// Function: sub_20C7CE0
// Address: 0x20c7ce0
//
__int64 __fastcall sub_20C7CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // r9
  unsigned __int64 v18; // rcx
  unsigned int v19; // edx
  int v20; // r9d
  unsigned __int8 v21; // al
  __int64 v22; // r8
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rbx
  int v29; // r12d
  __int64 v30; // r13
  _QWORD *v31; // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int64 v36; // r8
  unsigned int v37; // esi
  int v38; // eax
  __int64 v39; // rax
  unsigned int v40; // esi
  int v41; // eax
  unsigned __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdi
  unsigned int v46; // edx
  char v47; // al
  _QWORD *v48; // rsi
  unsigned int v49; // eax
  _QWORD *v50; // r15
  __int64 v51; // rdx
  unsigned int v52; // r14d
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // [rsp+0h] [rbp-80h]
  __int64 v56; // [rsp+8h] [rbp-78h]
  __int64 v57; // [rsp+8h] [rbp-78h]
  __int64 v58; // [rsp+8h] [rbp-78h]
  __int64 v59; // [rsp+10h] [rbp-70h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  unsigned __int64 v61; // [rsp+10h] [rbp-70h]
  __int64 v62; // [rsp+10h] [rbp-70h]
  __int64 v63; // [rsp+20h] [rbp-60h]
  __int64 v64; // [rsp+20h] [rbp-60h]
  unsigned __int64 v65; // [rsp+20h] [rbp-60h]
  __int64 v66; // [rsp+20h] [rbp-60h]
  unsigned __int64 v67; // [rsp+20h] [rbp-60h]
  __int64 *v68; // [rsp+28h] [rbp-58h]
  unsigned __int64 v69; // [rsp+28h] [rbp-58h]
  unsigned __int64 v70; // [rsp+28h] [rbp-58h]
  __int64 v71; // [rsp+28h] [rbp-58h]
  unsigned __int64 v72; // [rsp+28h] [rbp-58h]
  __int64 v73; // [rsp+28h] [rbp-58h]
  __int64 v74; // [rsp+28h] [rbp-58h]
  __int64 v75; // [rsp+28h] [rbp-58h]
  __int64 *v76; // [rsp+30h] [rbp-50h]
  int v77; // [rsp+30h] [rbp-50h]
  __int64 v78; // [rsp+30h] [rbp-50h]
  __int64 v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+30h] [rbp-50h]
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+38h] [rbp-48h]
  __int64 v84; // [rsp+38h] [rbp-48h]
  __int64 v85; // [rsp+38h] [rbp-48h]
  char v86[8]; // [rsp+40h] [rbp-40h] BYREF
  __int64 v87; // [rsp+48h] [rbp-38h]

  result = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)result == 13 )
  {
    v82 = sub_15A9930(a2, a3);
    result = *(_QWORD *)(a3 + 16);
    v76 = (__int64 *)(result + 8LL * *(unsigned int *)(a3 + 12));
    if ( (__int64 *)result != v76 )
    {
      v68 = *(__int64 **)(a3 + 16);
      v63 = a6;
      v12 = v68;
      do
      {
        v13 = *v12;
        v14 = (unsigned int)(v12 - v68);
        ++v12;
        result = sub_20C7CE0(a1, a2, v13, a4, a5, *(_QWORD *)(v82 + 8 * v14 + 16) + v63);
      }
      while ( v76 != v12 );
    }
  }
  else if ( (_BYTE)result == 14 )
  {
    v83 = *(_QWORD *)(a3 + 24);
    v15 = sub_15A9FE0(a2, v83);
    v16 = v83;
    v17 = 1;
    v18 = v15;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v16 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v32 = *(_QWORD *)(v16 + 32);
          v16 = *(_QWORD *)(v16 + 24);
          v17 *= v32;
          continue;
        case 1:
          v25 = 16;
          goto LABEL_27;
        case 2:
          v25 = 32;
          goto LABEL_27;
        case 3:
        case 9:
          v25 = 64;
          goto LABEL_27;
        case 4:
          v25 = 80;
          goto LABEL_27;
        case 5:
        case 6:
          v25 = 128;
          goto LABEL_27;
        case 7:
          v72 = v18;
          v37 = 0;
          v80 = v17;
          goto LABEL_39;
        case 0xB:
          v25 = *(_DWORD *)(v16 + 8) >> 8;
          goto LABEL_27;
        case 0xD:
          v70 = v18;
          v78 = v17;
          v31 = (_QWORD *)sub_15A9930(a2, v16);
          v17 = v78;
          v18 = v70;
          v25 = 8LL * *v31;
          goto LABEL_27;
        case 0xE:
          v59 = v18;
          v64 = v17;
          v71 = *(_QWORD *)(v16 + 24);
          v79 = *(_QWORD *)(v16 + 32);
          v33 = sub_15A9FE0(a2, v71);
          v18 = v59;
          v34 = v71;
          v35 = 1;
          v17 = v64;
          v36 = v33;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v34 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v44 = *(_QWORD *)(v34 + 32);
                v34 = *(_QWORD *)(v34 + 24);
                v35 *= v44;
                continue;
              case 1:
                v39 = 16;
                goto LABEL_45;
              case 2:
                v39 = 32;
                goto LABEL_45;
              case 3:
              case 9:
                v39 = 64;
                goto LABEL_45;
              case 4:
                v39 = 80;
                goto LABEL_45;
              case 5:
              case 6:
                v39 = 128;
                goto LABEL_45;
              case 7:
                v56 = v59;
                v40 = 0;
                v60 = v64;
                v65 = v36;
                v73 = v35;
                goto LABEL_50;
              case 0xB:
                v39 = *(_DWORD *)(v34 + 8) >> 8;
                goto LABEL_45;
              case 0xD:
                v58 = v59;
                v62 = v64;
                v67 = v36;
                v75 = v35;
                v43 = (_QWORD *)sub_15A9930(a2, v34);
                v35 = v75;
                v36 = v67;
                v17 = v62;
                v18 = v58;
                v39 = 8LL * *v43;
                goto LABEL_45;
              case 0xE:
                v55 = v59;
                v57 = v64;
                v61 = v36;
                v66 = v35;
                v74 = *(_QWORD *)(v34 + 32);
                v42 = sub_12BE0A0(a2, *(_QWORD *)(v34 + 24));
                v35 = v66;
                v36 = v61;
                v17 = v57;
                v18 = v55;
                v39 = 8 * v74 * v42;
                goto LABEL_45;
              case 0xF:
                v56 = v59;
                v60 = v64;
                v65 = v36;
                v40 = *(_DWORD *)(v34 + 8) >> 8;
                v73 = v35;
LABEL_50:
                v41 = sub_15A9520(a2, v40);
                v35 = v73;
                v36 = v65;
                v17 = v60;
                v18 = v56;
                v39 = (unsigned int)(8 * v41);
LABEL_45:
                v25 = 8 * v79 * v36 * ((v36 + ((unsigned __int64)(v39 * v35 + 7) >> 3) - 1) / v36);
                break;
            }
            goto LABEL_27;
          }
        case 0xF:
          v72 = v18;
          v80 = v17;
          v37 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_39:
          v38 = sub_15A9520(a2, v37);
          v17 = v80;
          v18 = v72;
          v25 = (unsigned int)(8 * v38);
LABEL_27:
          v69 = v18 * ((v18 + ((unsigned __int64)(v17 * v25 + 7) >> 3) - 1) / v18);
          result = *(_QWORD *)(a3 + 32);
          v77 = result;
          if ( (_DWORD)result )
          {
            v26 = a4;
            v27 = a6;
            v28 = a5;
            v29 = 0;
            v30 = v27;
            do
            {
              ++v29;
              result = sub_20C7CE0(a1, a2, v83, v26, v28, v30);
              v30 += v69;
            }
            while ( v77 != v29 );
          }
          break;
      }
      break;
    }
  }
  else if ( (_BYTE)result )
  {
    if ( (_BYTE)result == 15 )
    {
      v19 = 8 * sub_15A9520(a2, *(_DWORD *)(a3 + 8) >> 8);
      if ( v19 == 32 )
      {
        v21 = 5;
      }
      else if ( v19 > 0x20 )
      {
        v21 = 6;
        if ( v19 != 64 )
        {
          v21 = 0;
          if ( v19 == 128 )
            v21 = 7;
        }
      }
      else
      {
        v21 = 3;
        if ( v19 != 8 )
          v21 = 4 * (v19 == 16);
      }
      v22 = 0;
    }
    else if ( (_BYTE)result == 16 )
    {
      v45 = *(_QWORD *)(a3 + 24);
      if ( *(_BYTE *)(v45 + 8) == 15 )
      {
        v46 = 8 * sub_15A9520(a2, *(_DWORD *)(v45 + 8) >> 8);
        if ( v46 == 32 )
        {
          v47 = 5;
        }
        else if ( v46 > 0x20 )
        {
          v47 = 6;
          if ( v46 != 64 )
          {
            v47 = 0;
            if ( v46 == 128 )
              v47 = 7;
          }
        }
        else
        {
          v47 = 3;
          if ( v46 != 8 )
            v47 = 4 * (v46 == 16);
        }
        v48 = *(_QWORD **)a3;
        v86[0] = v47;
        v87 = 0;
        v45 = sub_1F58E60((__int64)v86, v48);
      }
      v84 = *(_QWORD *)(a3 + 32);
      LOBYTE(v49) = sub_1F59570(v45);
      v50 = *(_QWORD **)a3;
      v81 = v51;
      v52 = v49;
      v21 = sub_1D15020(v49, v84);
      v22 = 0;
      if ( !v21 )
      {
        v21 = sub_1F593D0(v50, v52, v81, v84);
        v22 = v53;
      }
    }
    else
    {
      v21 = sub_1F59570(a3);
      v22 = v54;
    }
    v23 = v21;
    v24 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v24 >= *(_DWORD *)(a4 + 12) )
    {
      v85 = v22;
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v22, v20);
      v24 = *(unsigned int *)(a4 + 8);
      v22 = v85;
    }
    result = *(_QWORD *)a4 + 16 * v24;
    *(_QWORD *)result = v23;
    *(_QWORD *)(result + 8) = v22;
    ++*(_DWORD *)(a4 + 8);
    if ( a5 )
    {
      result = *(unsigned int *)(a5 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a5 + 12) )
      {
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v22, v20);
        result = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * result) = a6;
      ++*(_DWORD *)(a5 + 8);
    }
  }
  return result;
}
