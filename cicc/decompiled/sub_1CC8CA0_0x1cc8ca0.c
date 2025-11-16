// Function: sub_1CC8CA0
// Address: 0x1cc8ca0
//
unsigned __int64 *__fastcall sub_1CC8CA0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        _BYTE *a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8,
        _QWORD *a9,
        _QWORD *a10)
{
  int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int8 v18; // al
  _QWORD *v19; // r8
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rcx
  _QWORD *v23; // rdi
  unsigned __int64 *v24; // rsi
  unsigned __int64 *v25; // rbx
  __int64 v26; // r12
  unsigned __int64 v27; // r13
  unsigned __int64 *v28; // rax
  __int64 *v30; // r13
  __int64 *v31; // rax
  unsigned __int64 v32; // rsi
  bool v33; // zf
  _QWORD *v34; // r8
  _QWORD *v35; // rax
  _QWORD *v36; // rsi
  _QWORD *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r13
  _QWORD *v40; // rdx
  unsigned __int64 *v41; // rsi
  _QWORD *v42; // r8
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  _QWORD *v48; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+30h] [rbp-70h]
  _QWORD *v52; // [rsp+40h] [rbp-60h]
  unsigned __int64 *v53; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v54; // [rsp+58h] [rbp-48h] BYREF
  unsigned __int64 v55; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 *v56[7]; // [rsp+68h] [rbp-38h] BYREF

  v10 = *(_DWORD *)(a2 + 20);
  v53 = a3;
  v11 = v10 & 0xFFFFFFF;
  if ( v11 )
  {
    v12 = a2;
    v14 = 0;
    v15 = 24LL * v11;
    v52 = a8 + 1;
    v50 = a1 + 112;
    v48 = a9 + 1;
    do
    {
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        v16 = *(_QWORD *)(v12 - 8);
      else
        v16 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
      v17 = *(_QWORD *)(v16 + v14);
      v18 = *(_BYTE *)(v17 + 16);
      if ( v18 <= 0x17u )
        goto LABEL_6;
      v54 = v17;
      if ( a6 != *(_QWORD *)(v17 + 40) || v18 == 77 )
        goto LABEL_6;
      v19 = a8 + 1;
      v20 = (_QWORD *)a8[2];
      if ( !v20 )
        goto LABEL_17;
      do
      {
        while ( 1 )
        {
          v21 = v20[2];
          v22 = v20[3];
          if ( v20[4] >= v17 )
            break;
          v20 = (_QWORD *)v20[3];
          if ( !v22 )
            goto LABEL_15;
        }
        v19 = v20;
        v20 = (_QWORD *)v20[2];
      }
      while ( v21 );
LABEL_15:
      if ( v52 == v19 || v19[4] > v17 )
      {
LABEL_17:
        v56[0] = &v54;
        v19 = (_QWORD *)sub_1CC72C0(a8, v19, v56);
      }
      if ( v19[5]
        || (unsigned __int8)sub_15F3040(v54)
        || !(unsigned __int8)sub_1CC8920(a1, v54, (unsigned __int64)v53, a7, a9, a10) )
      {
        goto LABEL_6;
      }
      v23 = (_QWORD *)v54;
      v24 = v53;
      if ( *(_QWORD *)(v54 + 8) )
      {
        v46 = v14;
        v25 = v53;
        v45 = v12;
        v26 = *(_QWORD *)(v54 + 8);
        v27 = v53[5];
        do
        {
          v28 = sub_1648700(v26);
          if ( *((_BYTE *)v28 + 16) > 0x17u
            && v25 != v28
            && v27 == v28[5]
            && !(unsigned __int8)sub_1CC8170(v50, (__int64)v25, (__int64)v28) )
          {
            v14 = v46;
            v12 = v45;
            goto LABEL_6;
          }
          v26 = *(_QWORD *)(v26 + 8);
        }
        while ( v26 );
        v14 = v46;
        v12 = v45;
        v23 = (_QWORD *)v54;
        v24 = v53;
      }
      sub_15F22F0(v23, (__int64)v24);
      v55 = v54;
      v56[0] = v53;
      v30 = sub_1467480(v50, (__int64 *)v56);
      v31 = sub_1467480(v50, (__int64 *)&v55);
      v32 = v54;
      v33 = v54 == a5;
      *((_DWORD *)v31 + 2) = *((_DWORD *)v30 + 2);
      if ( v33 )
      {
        v36 = a8 + 1;
        v37 = (_QWORD *)a8[2];
        if ( !v37 )
          goto LABEL_68;
        do
        {
          if ( v37[4] < a5 )
          {
            v37 = (_QWORD *)v37[3];
          }
          else
          {
            v36 = v37;
            v37 = (_QWORD *)v37[2];
          }
        }
        while ( v37 );
        if ( v52 == v36 || v36[4] > a5 )
        {
LABEL_68:
          v56[0] = &v54;
          v36 = (_QWORD *)sub_1CC72C0(a8, v36, v56);
        }
        v36[5] = 0;
      }
      else
      {
        v34 = a8 + 1;
        v35 = (_QWORD *)a8[2];
        if ( !v35 )
          goto LABEL_46;
        do
        {
          if ( v35[4] < v32 )
          {
            v35 = (_QWORD *)v35[3];
          }
          else
          {
            v34 = v35;
            v35 = (_QWORD *)v35[2];
          }
        }
        while ( v35 );
        if ( v52 == v34 || v34[4] > v32 )
        {
LABEL_46:
          v56[0] = &v54;
          v34 = (_QWORD *)sub_1CC72C0(a8, v34, v56);
        }
        v34[5] = a5;
      }
      v38 = (_QWORD *)a9[2];
      if ( v38 )
      {
        v39 = a9 + 1;
        v40 = (_QWORD *)a9[2];
        do
        {
          if ( v40[4] < (unsigned __int64)v53 )
          {
            v40 = (_QWORD *)v40[3];
          }
          else
          {
            v39 = v40;
            v40 = (_QWORD *)v40[2];
          }
        }
        while ( v40 );
        if ( v39 != v48 && v39[4] <= (unsigned __int64)v53 )
        {
LABEL_57:
          v41 = (unsigned __int64 *)v54;
          v42 = a9 + 1;
          do
          {
            if ( v38[4] < v54 )
            {
              v38 = (_QWORD *)v38[3];
            }
            else
            {
              v42 = v38;
              v38 = (_QWORD *)v38[2];
            }
          }
          while ( v38 );
          if ( v42 != v48 && v42[4] <= v54 )
            goto LABEL_65;
          goto LABEL_64;
        }
      }
      else
      {
        v39 = a9 + 1;
      }
      v56[0] = (unsigned __int64 *)&v53;
      v39 = (_QWORD *)sub_1CC72C0(a9, v39, v56);
      v38 = (_QWORD *)a9[2];
      if ( v38 )
        goto LABEL_57;
      v42 = a9 + 1;
LABEL_64:
      v56[0] = &v54;
      v43 = sub_1CC72C0(a9, v42, v56);
      v41 = (unsigned __int64 *)v54;
      v42 = (_QWORD *)v43;
LABEL_65:
      v44 = v39[5];
      v53 = v41;
      v42[5] = v44;
      *a4 = 1;
LABEL_6:
      v14 += 24;
    }
    while ( v15 != v14 );
  }
  return v53;
}
