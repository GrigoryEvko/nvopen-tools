// Function: sub_32693F0
// Address: 0x32693f0
//
__int64 __fastcall sub_32693F0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 *v5; // rax
  __int64 (*v6)(); // r10
  __int64 result; // rax
  __int64 v8; // r15
  char v9; // al
  __int64 v10; // r8
  unsigned int v11; // ecx
  __int64 v12; // r9
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rax
  int v16; // eax
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 v22; // r11
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int16 *v25; // rax
  int v26; // ebx
  __int128 v27; // rax
  int v28; // r9d
  __int64 v29; // rax
  int v30; // edx
  int v31; // eax
  __int64 v32; // rax
  __int128 v33; // [rsp-20h] [rbp-B0h]
  __int128 v34; // [rsp-10h] [rbp-A0h]
  __int128 v35; // [rsp-10h] [rbp-A0h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  unsigned int v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  unsigned int v39; // [rsp+30h] [rbp-60h]
  __int64 v40; // [rsp+30h] [rbp-60h]
  __int64 v41; // [rsp+30h] [rbp-60h]
  __int64 v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+30h] [rbp-60h]
  __int64 v44; // [rsp+38h] [rbp-58h]
  __int64 v45; // [rsp+40h] [rbp-50h]
  int v46; // [rsp+40h] [rbp-50h]
  __int64 v47; // [rsp+48h] [rbp-48h]
  int v48; // [rsp+48h] [rbp-48h]
  __int64 v49; // [rsp+48h] [rbp-48h]
  __int64 v50; // [rsp+50h] [rbp-40h] BYREF
  int v51; // [rsp+58h] [rbp-38h]

  v4 = a1[1];
  v5 = *(__int64 **)(a2 + 40);
  v6 = *(__int64 (**)())(*(_QWORD *)v4 + 408LL);
  if ( v6 == sub_2FE3040 )
    return 0;
  v45 = *v5;
  v8 = *((unsigned int *)v5 + 12);
  v39 = *((_DWORD *)v5 + 2);
  v47 = v5[5];
  v9 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v6)(v4, *v5, v5[1]);
  v10 = v47;
  v11 = v39;
  v12 = v45;
  if ( !v9 )
    return 0;
  v13 = *(_QWORD *)(v47 + 56);
  if ( !v13 )
    goto LABEL_23;
  v14 = 1;
  do
  {
    while ( (_DWORD)v8 != *(_DWORD *)(v13 + 8) )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_13;
    }
    if ( !v14 )
      goto LABEL_23;
    v15 = *(_QWORD *)(v13 + 32);
    if ( !v15 )
      goto LABEL_14;
    if ( (_DWORD)v8 == *(_DWORD *)(v15 + 8) )
      goto LABEL_23;
    v13 = *(_QWORD *)(v15 + 32);
    v14 = 0;
  }
  while ( v13 );
LABEL_13:
  if ( v14 == 1 )
    goto LABEL_23;
LABEL_14:
  v16 = *(_DWORD *)(v47 + 24);
  v48 = v16;
  if ( v16 == 190 )
  {
    v46 = 192;
  }
  else
  {
    if ( v16 != 192 )
      goto LABEL_23;
    v46 = 190;
  }
  v36 = v12;
  v37 = v39;
  v40 = v10;
  v17 = sub_33CF460(**(_QWORD **)(v10 + 40), *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL));
  v10 = v40;
  v11 = v37;
  v12 = v36;
  if ( !v17 )
  {
LABEL_23:
    v29 = *(_QWORD *)(v12 + 56);
    if ( v29 )
    {
      v30 = 1;
      do
      {
        if ( v11 == *(_DWORD *)(v29 + 8) )
        {
          if ( !v30 )
            return 0;
          v29 = *(_QWORD *)(v29 + 32);
          if ( !v29 )
            goto LABEL_33;
          if ( v11 == *(_DWORD *)(v29 + 8) )
            return 0;
          v30 = 0;
        }
        v29 = *(_QWORD *)(v29 + 32);
      }
      while ( v29 );
      if ( v30 == 1 )
        return 0;
LABEL_33:
      v31 = *(_DWORD *)(v12 + 24);
      v48 = v31;
      if ( v31 == 190 )
      {
        v46 = 192;
      }
      else
      {
        if ( v31 != 192 )
          return 0;
        v46 = 190;
      }
      v38 = v10;
      v43 = v12;
      if ( (unsigned __int8)sub_33CF460(**(_QWORD **)(v12 + 40), *(_QWORD *)(*(_QWORD *)(v12 + 40) + 8LL)) )
      {
        LODWORD(v12) = v43;
        v32 = *(_QWORD *)(v43 + 40);
        v19 = v38;
        v20 = *(_QWORD *)(v32 + 40);
        v21 = *(unsigned int *)(v32 + 48);
        v22 = v8;
        goto LABEL_19;
      }
    }
    return 0;
  }
  v18 = *(_QWORD *)(v40 + 40);
  v19 = v36;
  v20 = *(_QWORD *)(v18 + 40);
  v21 = *(unsigned int *)(v18 + 48);
  v22 = v37;
LABEL_19:
  v23 = *(_QWORD *)(a2 + 80);
  v50 = v23;
  if ( v23 )
  {
    v41 = v19;
    v44 = v22;
    sub_B96E90((__int64)&v50, v23, 1);
    v19 = v41;
    v22 = v44;
  }
  v24 = *a1;
  v51 = *(_DWORD *)(a2 + 72);
  v25 = *(unsigned __int16 **)(a2 + 48);
  v26 = *v25;
  *((_QWORD *)&v34 + 1) = v21;
  *(_QWORD *)&v34 = v20;
  *((_QWORD *)&v33 + 1) = v22;
  *(_QWORD *)&v33 = v19;
  v42 = *((_QWORD *)v25 + 1);
  *(_QWORD *)&v27 = sub_3406EB0(v24, v46, (unsigned int)&v50, v26, v42, v12, v33, v34);
  *((_QWORD *)&v35 + 1) = v21;
  *(_QWORD *)&v35 = v20;
  result = sub_3406EB0(*a1, v48, (unsigned int)&v50, v26, v42, v28, v27, v35);
  if ( v50 )
  {
    v49 = result;
    sub_B91220((__int64)&v50, v50);
    return v49;
  }
  return result;
}
