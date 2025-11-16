// Function: sub_23D3BD0
// Address: 0x23d3bd0
//
__int64 __fastcall sub_23D3BD0(__int64 a1, _BYTE *a2)
{
  char v3; // al
  __int64 result; // rax
  _BYTE *v5; // rcx
  _BYTE *v6; // r13
  _BYTE *v7; // rdx
  _BYTE *v8; // r14
  __int64 v9; // r8
  unsigned int v10; // r15d
  int v11; // eax
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  unsigned int v15; // r14d
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  unsigned int v19; // r13d
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  unsigned int v23; // r14d
  int v24; // eax
  bool v25; // al
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r14
  _BYTE *v29; // rax
  unsigned int v30; // r14d
  int v31; // eax
  __int64 v32; // r15
  __int64 v33; // rdx
  _BYTE *v34; // rax
  unsigned int v35; // r15d
  int v36; // eax
  bool v37; // al
  int v38; // eax
  bool v39; // r15
  unsigned int v40; // r14d
  __int64 v41; // rax
  unsigned int v42; // r15d
  int v43; // eax
  int v44; // eax
  __int64 v45; // rsi
  bool v46; // r15
  __int64 v47; // rax
  unsigned int v48; // r15d
  int v49; // eax
  _BYTE *v50; // [rsp+0h] [rbp-50h]
  __int64 v51; // [rsp+0h] [rbp-50h]
  _BYTE *v52; // [rsp+0h] [rbp-50h]
  _BYTE *v53; // [rsp+8h] [rbp-48h]
  _BYTE *v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+8h] [rbp-48h]
  _BYTE *v56; // [rsp+8h] [rbp-48h]
  _BYTE *v57; // [rsp+8h] [rbp-48h]
  _BYTE *v58; // [rsp+10h] [rbp-40h]
  _BYTE *v59; // [rsp+10h] [rbp-40h]
  _BYTE *v60; // [rsp+10h] [rbp-40h]
  _BYTE *v61; // [rsp+10h] [rbp-40h]
  __int64 v62; // [rsp+10h] [rbp-40h]
  __int64 v63; // [rsp+10h] [rbp-40h]
  _BYTE *v64; // [rsp+10h] [rbp-40h]
  __int64 v65; // [rsp+10h] [rbp-40h]
  __int64 v66; // [rsp+18h] [rbp-38h]
  __int64 v67; // [rsp+18h] [rbp-38h]
  __int64 v68; // [rsp+18h] [rbp-38h]
  _BYTE *v69; // [rsp+18h] [rbp-38h]
  _BYTE *v70; // [rsp+18h] [rbp-38h]
  int v71; // [rsp+18h] [rbp-38h]
  int v72; // [rsp+18h] [rbp-38h]

  v3 = *a2;
  if ( *a2 != 68 )
  {
LABEL_2:
    if ( v3 != 55 )
      return 0;
    v5 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v5 != 46 )
      return 0;
    v6 = (_BYTE *)*((_QWORD *)v5 - 8);
    if ( *v6 != 57 )
      return 0;
    v7 = (_BYTE *)*((_QWORD *)v6 - 8);
    if ( *v7 != 44 )
      goto LABEL_8;
    v22 = *((_QWORD *)v7 - 8);
    if ( *(_BYTE *)v22 == 17 )
    {
      v23 = *(_DWORD *)(v22 + 32);
      if ( v23 <= 0x40 )
      {
        v25 = *(_QWORD *)(v22 + 24) == 0;
      }
      else
      {
        v53 = (_BYTE *)*((_QWORD *)v6 - 8);
        v59 = (_BYTE *)*((_QWORD *)a2 - 8);
        v67 = *((_QWORD *)v7 - 8);
        v24 = sub_C444A0(v22 + 24);
        v22 = v67;
        v5 = v59;
        v7 = v53;
        v25 = v23 == v24;
      }
    }
    else
    {
      v28 = *(_QWORD *)(v22 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 > 1 || *(_BYTE *)v22 > 0x15u )
        goto LABEL_8;
      v54 = (_BYTE *)*((_QWORD *)v6 - 8);
      v60 = (_BYTE *)*((_QWORD *)a2 - 8);
      v68 = *((_QWORD *)v7 - 8);
      v29 = sub_AD7630(v68, 0, (__int64)v7);
      v22 = v68;
      v5 = v60;
      v7 = v54;
      if ( !v29 || *v29 != 17 )
      {
        if ( *(_BYTE *)(v28 + 8) != 17 )
          goto LABEL_8;
        v38 = *(_DWORD *)(v28 + 32);
        v39 = 0;
        v40 = 0;
        v71 = v38;
        while ( v71 != v40 )
        {
          v50 = v7;
          v56 = v5;
          v63 = v22;
          v41 = sub_AD69F0((unsigned __int8 *)v22, v40);
          v22 = v63;
          v5 = v56;
          v7 = v50;
          if ( !v41 )
            goto LABEL_8;
          if ( *(_BYTE *)v41 != 13 )
          {
            if ( *(_BYTE *)v41 != 17 )
              goto LABEL_8;
            v42 = *(_DWORD *)(v41 + 32);
            if ( v42 <= 0x40 )
            {
              v39 = *(_QWORD *)(v41 + 24) == 0;
            }
            else
            {
              v51 = v63;
              v57 = v7;
              v64 = v5;
              v43 = sub_C444A0(v41 + 24);
              v5 = v64;
              v7 = v57;
              v22 = v51;
              v39 = v42 == v43;
            }
            if ( !v39 )
              goto LABEL_8;
          }
          ++v40;
        }
        if ( !v39 )
          goto LABEL_8;
LABEL_34:
        v26 = *(__int64 **)(a1 + 40);
        if ( v26 )
          *v26 = v22;
        v27 = *((_QWORD *)v7 - 4);
        if ( v27 )
        {
          **(_QWORD **)(a1 + 48) = v27;
          v8 = (_BYTE *)*((_QWORD *)v6 - 4);
          if ( v8 == **(_BYTE ***)(a1 + 56) )
          {
LABEL_17:
            v14 = *((_QWORD *)v5 - 4);
            if ( *(_BYTE *)v14 == 17 )
            {
              v15 = *(_DWORD *)(v14 + 32);
              if ( v15 <= 0x40 )
              {
                v16 = *(_QWORD **)(a1 + 64);
                v17 = *(_QWORD *)(v14 + 24);
                goto LABEL_20;
              }
              if ( v15 - (unsigned int)sub_C444A0(v14 + 24) <= 0x40 )
              {
                v16 = *(_QWORD **)(a1 + 64);
                v17 = **(_QWORD **)(v14 + 24);
LABEL_20:
                *v16 = v17;
                v18 = *((_QWORD *)a2 - 4);
                if ( *(_BYTE *)v18 == 17 )
                {
                  v19 = *(_DWORD *)(v18 + 32);
                  if ( v19 <= 0x40 )
                  {
                    v20 = *(_QWORD **)(a1 + 72);
                    v21 = *(_QWORD *)(v18 + 24);
LABEL_23:
                    *v20 = v21;
                    return 1;
                  }
                  if ( v19 - (unsigned int)sub_C444A0(v18 + 24) <= 0x40 )
                  {
                    v20 = *(_QWORD **)(a1 + 72);
                    v21 = **(_QWORD **)(v18 + 24);
                    goto LABEL_23;
                  }
                }
              }
            }
            return 0;
          }
LABEL_9:
          if ( *v8 != 44 )
            return 0;
          v9 = *((_QWORD *)v8 - 8);
          if ( *(_BYTE *)v9 == 17 )
          {
            v10 = *(_DWORD *)(v9 + 32);
            if ( v10 > 0x40 )
            {
              v58 = v5;
              v66 = *((_QWORD *)v8 - 8);
              v11 = sub_C444A0(v9 + 24);
              v9 = v66;
              v5 = v58;
              if ( v10 != v11 )
                return 0;
              goto LABEL_13;
            }
            v37 = *(_QWORD *)(v9 + 24) == 0;
          }
          else
          {
            v32 = *(_QWORD *)(v9 + 8);
            v70 = v5;
            v33 = (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17;
            if ( (unsigned int)v33 > 1 || *(_BYTE *)v9 > 0x15u )
              return 0;
            v62 = *((_QWORD *)v8 - 8);
            v34 = sub_AD7630(v9, 0, v33);
            v9 = v62;
            v5 = v70;
            if ( !v34 || *v34 != 17 )
            {
              if ( *(_BYTE *)(v32 + 8) != 17 )
                return 0;
              v44 = *(_DWORD *)(v32 + 32);
              v45 = 0;
              v46 = 0;
              v72 = v44;
              while ( v72 != (_DWORD)v45 )
              {
                v52 = v5;
                v65 = v9;
                v47 = sub_AD69F0((unsigned __int8 *)v9, v45);
                if ( !v47 )
                  return 0;
                v9 = v65;
                v5 = v52;
                if ( *(_BYTE *)v47 != 13 )
                {
                  if ( *(_BYTE *)v47 != 17 )
                    return 0;
                  v48 = *(_DWORD *)(v47 + 32);
                  if ( v48 <= 0x40 )
                  {
                    v46 = *(_QWORD *)(v47 + 24) == 0;
                  }
                  else
                  {
                    v49 = sub_C444A0(v47 + 24);
                    v5 = v52;
                    v9 = v65;
                    v46 = v48 == v49;
                  }
                  if ( !v46 )
                    return 0;
                }
                v45 = (unsigned int)(v45 + 1);
              }
              if ( !v46 )
                return 0;
LABEL_13:
              v12 = *(__int64 **)(a1 + 40);
              if ( v12 )
                *v12 = v9;
              v13 = *((_QWORD *)v8 - 4);
              if ( !v13 )
                return 0;
              **(_QWORD **)(a1 + 48) = v13;
              if ( *((_QWORD *)v6 - 8) != **(_QWORD **)(a1 + 56) )
                return 0;
              goto LABEL_17;
            }
            v35 = *((_DWORD *)v34 + 8);
            if ( v35 <= 0x40 )
            {
              v37 = *((_QWORD *)v34 + 3) == 0;
            }
            else
            {
              v36 = sub_C444A0((__int64)(v34 + 24));
              v5 = v70;
              v9 = v62;
              v37 = v35 == v36;
            }
          }
          if ( !v37 )
            return 0;
          goto LABEL_13;
        }
LABEL_8:
        v8 = (_BYTE *)*((_QWORD *)v6 - 4);
        goto LABEL_9;
      }
      v30 = *((_DWORD *)v29 + 8);
      if ( v30 <= 0x40 )
      {
        v25 = *((_QWORD *)v29 + 3) == 0;
      }
      else
      {
        v55 = v68;
        v61 = v7;
        v69 = v5;
        v31 = sub_C444A0((__int64)(v29 + 24));
        v5 = v69;
        v7 = v61;
        v22 = v55;
        v25 = v30 == v31;
      }
    }
    if ( !v25 )
      goto LABEL_8;
    goto LABEL_34;
  }
  result = sub_23D36F0(a1, 26, *((unsigned __int8 **)a2 - 4));
  if ( !(_BYTE)result )
  {
    v3 = *a2;
    goto LABEL_2;
  }
  return result;
}
