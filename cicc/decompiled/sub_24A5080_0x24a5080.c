// Function: sub_24A5080
// Address: 0x24a5080
//
__int64 __fastcall sub_24A5080(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  int v12; // edx
  unsigned __int64 v13; // rax
  bool v14; // cf
  __int64 v15; // rdx
  __int64 *v16; // rdi
  unsigned int v17; // r12d
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // esi
  __int64 v21; // r9
  int v22; // esi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rdi
  __int64 *v26; // r10
  __int64 v27; // r10
  __int64 v28; // rdx
  __int64 v29; // rsi
  char v30; // al
  unsigned __int64 v31; // rcx
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  int v34; // eax
  unsigned __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 v38; // r9
  __int64 v39; // rdi
  __int64 v40; // rdi
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  __int64 v43; // r11
  unsigned int v44; // ecx
  int v45; // r10d
  int v46; // r8d
  int i; // eax
  int v48; // ecx
  __int64 v49; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+10h] [rbp-C0h]
  __int64 v51; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v52; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v53; // [rsp+20h] [rbp-B0h]
  __int64 v54; // [rsp+28h] [rbp-A8h]
  __int64 v55; // [rsp+38h] [rbp-98h]
  __int64 v56; // [rsp+40h] [rbp-90h]
  __int64 v57; // [rsp+48h] [rbp-88h]
  int v58; // [rsp+54h] [rbp-7Ch]
  unsigned __int64 v59; // [rsp+58h] [rbp-78h]
  unsigned __int64 v60; // [rsp+60h] [rbp-70h]
  unsigned __int64 v61; // [rsp+68h] [rbp-68h]
  __int64 v62; // [rsp+70h] [rbp-60h]
  __int64 v63; // [rsp+78h] [rbp-58h]
  __int64 v64; // [rsp+78h] [rbp-58h]
  char v65; // [rsp+78h] [rbp-58h]
  unsigned __int64 v66; // [rsp+80h] [rbp-50h]
  __int64 v67; // [rsp+88h] [rbp-48h]
  unsigned int v68[13]; // [rsp+9Ch] [rbp-34h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(a1 + 80);
  v51 = 2;
  v4 = *(_QWORD *)(v2 + 80);
  v5 = v4 - 24;
  if ( !v4 )
    v5 = 0;
  v56 = v5;
  if ( v3 )
    v51 = sub_FDC4B0(v3);
  v6 = 0;
  if ( !*(_BYTE *)(a1 + 96) )
    v6 = v51;
  v52 = v6;
  v49 = sub_24A4AF0(a1, 0, v56, v6);
  v7 = *(_QWORD *)(v56 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == v56 + 48 )
    return sub_24A4AF0(a1, v56, 0, v52);
  if ( !v7 )
LABEL_55:
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA || !(unsigned int)sub_B46E30(v7 - 24) )
    return sub_24A4AF0(a1, v56, 0, v52);
  v54 = *(_QWORD *)a1 + 72LL;
  v55 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( v55 == v54 )
  {
    v62 = 0;
    v59 = 0;
    v66 = 0;
    v57 = 0;
LABEL_72:
    result = 3 * v59;
    if ( 2 * v66 < 3 * v59 )
    {
      *(_QWORD *)(v62 + 16) = v59;
      *(_QWORD *)(v57 + 16) = v66 + 1;
      return v66 + 1;
    }
    return result;
  }
  v59 = 0;
  v53 = 0;
  v66 = 0;
  v57 = 0;
  v50 = 0;
  v62 = 0;
  do
  {
    if ( !v55 )
      BUG();
    v10 = v55 - 24;
    v11 = *(_QWORD *)(v55 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == v55 + 24 )
    {
      v67 = 0;
    }
    else
    {
      if ( !v11 )
        BUG();
      v12 = *(unsigned __int8 *)(v11 - 24);
      v13 = v11 - 24;
      v14 = (unsigned int)(v12 - 30) < 0xB;
      v15 = 0;
      if ( v14 )
        v15 = v13;
      v67 = v15;
    }
    v16 = *(__int64 **)(a1 + 80);
    v61 = 2;
    if ( v16 )
      v61 = sub_FDD860(v16, v55 - 24);
    v58 = sub_B46E30(v67);
    if ( v58 )
    {
      v17 = 0;
      v18 = 2;
      while ( 1 )
      {
        v37 = sub_B46EC0(v67, v17);
        v65 = sub_D0E970(v67, v17, 0);
        v38 = v61;
        if ( v65 )
        {
          v38 = -1;
          if ( v61 <= 0x4189374BC6A7EELL )
            v38 = 1000 * v61;
        }
        v39 = *(_QWORD *)(a1 + 72);
        v60 = v38;
        if ( v39 )
        {
          v68[0] = sub_FF0430(v39, v10, v37);
          v18 = sub_F02E20(v68, v60);
          if ( !*(_BYTE *)(a1 + 97) )
            goto LABEL_31;
        }
        else if ( !*(_BYTE *)(a1 + 97) )
        {
          goto LABEL_33;
        }
        v19 = *(_QWORD *)(a1 + 88);
        v20 = *(_DWORD *)(v19 + 24);
        v21 = *(_QWORD *)(v19 + 8);
        if ( !v20 )
          goto LABEL_31;
        v22 = v20 - 1;
        v23 = v22 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v24 = (__int64 *)(v21 + 16LL * v23);
        v25 = *v24;
        v26 = v24;
        if ( v37 != *v24 )
        {
          v43 = *v24;
          v44 = v22 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v45 = 1;
          while ( v43 != -4096 )
          {
            v46 = v45 + 1;
            v44 = v22 & (v45 + v44);
            v26 = (__int64 *)(v21 + 16LL * v44);
            v43 = *v26;
            if ( v37 == *v26 )
              goto LABEL_29;
            v45 = v46;
          }
          goto LABEL_31;
        }
LABEL_29:
        v27 = v26[1];
        if ( !v27 || v37 != **(_QWORD **)(v27 + 32) )
          goto LABEL_31;
        if ( v37 != v25 )
        {
          for ( i = 1; ; i = v48 )
          {
            if ( v25 == -4096 )
              BUG();
            v48 = i + 1;
            v23 = v22 & (i + v23);
            v24 = (__int64 *)(v21 + 16LL * v23);
            v25 = *v24;
            if ( v37 == *v24 )
              break;
          }
        }
        v40 = v24[1];
        if ( *(_BYTE *)(v40 + 84) )
        {
          v41 = *(_QWORD **)(v40 + 64);
          v42 = &v41[*(unsigned int *)(v40 + 76)];
          if ( v41 != v42 )
          {
            while ( v10 != *v41 )
            {
              if ( v42 == ++v41 )
                goto LABEL_61;
            }
LABEL_31:
            if ( !v18 )
              v18 = 1;
            goto LABEL_33;
          }
LABEL_61:
          v18 = 1;
        }
        else
        {
          if ( sub_C8CA60(v40 + 56, v10) )
            goto LABEL_31;
          v18 = 1;
        }
LABEL_33:
        v28 = sub_24A4AF0(a1, v10, v37, v18);
        v29 = *(_QWORD *)(v28 + 8);
        *(_BYTE *)(v28 + 26) = v65;
        if ( v29 )
        {
          v63 = v28;
          v30 = sub_F35EF0(*(_QWORD *)v28, v29);
          v28 = v63;
          if ( v30 )
            *(_BYTE *)(v63 + 25) = 1;
        }
        v31 = v66;
        if ( v18 > v66 )
        {
          v32 = v62;
          if ( v56 == v10 )
          {
            v31 = v18;
            v32 = v28;
          }
          v66 = v31;
          v62 = v32;
        }
        v33 = *(_QWORD *)(v37 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v33 != v37 + 48 )
        {
          if ( !v33 )
            goto LABEL_55;
          v64 = v28;
          if ( (unsigned int)*(unsigned __int8 *)(v33 - 24) - 30 <= 0xA )
          {
            v34 = sub_B46E30(v33 - 24);
            v35 = v59;
            if ( v18 > v59 )
            {
              v36 = v64;
              if ( v34 )
                v36 = v57;
              else
                v35 = v18;
              v57 = v36;
              v59 = v35;
            }
          }
        }
        if ( v58 == ++v17 )
          goto LABEL_15;
      }
    }
    *(_BYTE *)(a1 + 64) = 1;
    v8 = sub_24A4AF0(a1, v10, 0, v61);
    if ( v61 > v53 )
    {
      v53 = v61;
      v50 = v8;
    }
LABEL_15:
    result = *(_QWORD *)(v55 + 8);
    v55 = result;
  }
  while ( v54 != result );
  if ( v52 < v53 || (result = 3 * v53, 2 * v52 >= 3 * v53) )
  {
    if ( v66 < v59 )
      return result;
    goto LABEL_72;
  }
  *(_QWORD *)(v49 + 16) = v53;
  result = v52 + 1;
  *(_QWORD *)(v50 + 16) = v52 + 1;
  if ( v66 >= v59 )
    goto LABEL_72;
  return result;
}
