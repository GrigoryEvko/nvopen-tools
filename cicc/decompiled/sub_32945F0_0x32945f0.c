// Function: sub_32945F0
// Address: 0x32945f0
//
__int64 __fastcall sub_32945F0(unsigned int *a1, unsigned int *a2, unsigned int a3, __int64 a4, int a5)
{
  unsigned int v8; // ebx
  unsigned int v9; // edx
  __int64 v10; // rax
  int v11; // ecx
  __int64 v12; // rax
  unsigned __int16 *v13; // rax
  int v14; // r13d
  __int128 v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rbx
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  __int16 v24; // dx
  __int64 v25; // rax
  __int64 v26; // r12
  unsigned int v27; // edx
  unsigned int v28; // r13d
  __int64 v29; // rbx
  unsigned int v30; // ecx
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // r9d
  __int64 v34; // rdi
  __int64 v35; // rdx
  unsigned int v36; // edx
  __int64 v37; // rax
  __int128 v38; // rax
  int v39; // r9d
  __int64 v40; // rcx
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // edx
  unsigned int v45; // edx
  __int128 v46; // [rsp-10h] [rbp-D0h]
  __int128 v47; // [rsp-10h] [rbp-D0h]
  __int128 v48; // [rsp+0h] [rbp-C0h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  __int64 v50; // [rsp+18h] [rbp-A8h]
  __int16 v51; // [rsp+26h] [rbp-9Ah]
  __int64 v53; // [rsp+70h] [rbp-50h] BYREF
  __int64 v54; // [rsp+78h] [rbp-48h]
  __int64 v55; // [rsp+80h] [rbp-40h] BYREF
  int v56; // [rsp+88h] [rbp-38h]

  if ( (_BYTE)a3 )
    return 0;
  v8 = a3;
  if ( !(unsigned __int8)sub_33CF170(*(_QWORD *)a1, *((_QWORD *)a1 + 1)) )
  {
    v9 = a2[2];
    v10 = *(_QWORD *)(*(_QWORD *)a2 + 56LL);
    if ( !v10 )
      return 0;
    v11 = 1;
    do
    {
      while ( v9 != *(_DWORD *)(v10 + 8) )
      {
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 )
          goto LABEL_12;
      }
      if ( !v11 )
        return 0;
      v12 = *(_QWORD *)(v10 + 32);
      if ( !v12 )
        goto LABEL_13;
      if ( v9 == *(_DWORD *)(v12 + 8) )
        return 0;
      v10 = *(_QWORD *)(v12 + 32);
      v11 = 0;
    }
    while ( v10 );
LABEL_12:
    if ( v11 == 1 )
      return 0;
  }
LABEL_13:
  v13 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a1 + 48LL) + 16LL * a1[2]);
  v51 = *v13;
  v14 = *v13;
  v50 = *((_QWORD *)v13 + 1);
  *(_QWORD *)&v15 = sub_3407D00(a4, *(_QWORD *)a2, *((_QWORD *)a2 + 1), 0);
  if ( (_QWORD)v15 )
  {
    v49 = v15;
    v48 = v15;
    if ( !(unsigned __int8)sub_33CF170(v15, *((_QWORD *)&v15 + 1)) )
    {
      v17 = *(_QWORD *)(v49 + 48) + 16LL * DWORD2(v48);
      if ( v51 == *(_WORD *)v17 && (*(_QWORD *)(v17 + 8) == v50 || v51) )
      {
        *(_QWORD *)a1 = sub_3406EB0(a4, 56, a5, v14, v50, v16, *(_OWORD *)a1, v48);
        a1[2] = v18;
        v19 = sub_3400BD0(a4, 0, a5, v14, v50, 0, 0);
        v22 = v21;
        v23 = *(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * a2[2];
        v24 = *(_WORD *)v23;
        v25 = *(_QWORD *)(v23 + 8);
        LOWORD(v53) = v24;
        v54 = v25;
        if ( v24 )
        {
          if ( (unsigned __int16)(v24 - 176) <= 0x34u )
          {
LABEL_20:
            if ( *(_DWORD *)(v19 + 24) == 51 )
            {
              v55 = 0;
              v56 = 0;
              v26 = sub_33F17F0(a4, 51, &v55, v53, v54);
              v28 = v45;
              if ( v55 )
                sub_B91220((__int64)&v55, v55);
            }
            else
            {
              *((_QWORD *)&v46 + 1) = v22;
              *(_QWORD *)&v46 = v19;
              v26 = sub_33FAF80(a4, 168, a5, v53, v54, v20, v46);
              v28 = v27;
            }
            v29 = v26;
            v30 = v28;
            goto LABEL_23;
          }
        }
        else if ( sub_3007100((__int64)&v53) )
        {
          goto LABEL_20;
        }
        v43 = sub_32886A0(a4, (unsigned int)v53, v54, a5, v19, v22);
        v30 = v44;
        v29 = v43;
LABEL_23:
        *(_QWORD *)a2 = v29;
        v8 = 1;
        a2[2] = v30;
        return v8;
      }
    }
  }
  if ( *(_DWORD *)(*(_QWORD *)a2 + 24LL) != 56 )
    return v8;
  v31 = sub_3407D00(a4, **(_QWORD **)(*(_QWORD *)a2 + 40LL), *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 40LL) + 8LL), 0);
  v34 = v32;
  if ( v31
    && (v35 = *(_QWORD *)(v31 + 48) + 16LL * (unsigned int)v32, v51 == *(_WORD *)v35)
    && (*(_QWORD *)(v35 + 8) == v50 || v51) )
  {
    *((_QWORD *)&v47 + 1) = v34;
    *(_QWORD *)&v47 = v31;
    *(_QWORD *)a1 = sub_3406EB0(a4, 56, a5, v14, v50, v33, *(_OWORD *)a1, v47);
    a1[2] = v36;
    v37 = *(_QWORD *)(*(_QWORD *)a2 + 40LL);
    *(_QWORD *)a2 = *(_QWORD *)(v37 + 40);
    a2[2] = *(_DWORD *)(v37 + 48);
    return 1;
  }
  else
  {
    *(_QWORD *)&v38 = sub_3407D00(
                        a4,
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 40LL) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 40LL) + 48LL),
                        0);
    if ( !(_QWORD)v38 )
      return v8;
    v40 = *(_QWORD *)(v38 + 48) + 16LL * DWORD2(v38);
    if ( v51 != *(_WORD *)v40 || *(_QWORD *)(v40 + 8) != v50 && !v51 )
      return v8;
    *(_QWORD *)a1 = sub_3406EB0(a4, 56, a5, v14, v50, v39, *(_OWORD *)a1, v38);
    a1[2] = v41;
    v42 = *(_QWORD *)(*(_QWORD *)a2 + 40LL);
    *(_QWORD *)a2 = *(_QWORD *)v42;
    a2[2] = *(_DWORD *)(v42 + 8);
    return 1;
  }
}
