// Function: sub_1C620D0
// Address: 0x1c620d0
//
__int64 __fastcall sub_1C620D0(
        _QWORD *a1,
        unsigned int *a2,
        __int64 **a3,
        _QWORD *a4,
        unsigned __int64 *a5,
        __int64 a6,
        _BYTE *a7)
{
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r13
  unsigned __int64 v15; // r12
  unsigned int v16; // eax
  unsigned int v17; // r15d
  int v18; // r14d
  __int64 v19; // rax
  int v20; // esi
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 *v23; // rdi
  _QWORD *v24; // r8
  __int64 *v25; // rdx
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // r8
  unsigned __int64 *v29; // rax
  __int64 v31; // rsi
  unsigned int v32; // eax
  int v33; // eax
  int v34; // r9d
  int v35; // edi
  int v36; // r9d
  unsigned __int64 v37; // r14
  __int64 v38; // rax
  _QWORD *v39; // r15
  _QWORD *v40; // rax
  __int64 v41; // rdi
  _QWORD *v42; // rax
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rsi
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v52[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(__int64 **)a2;
  v10 = **(_QWORD **)a2;
  v11 = **a3;
  v12 = *(_QWORD *)(v10 + 16);
  v46 = v10;
  v13 = *(_QWORD *)(v11 + 16);
  v47 = v11;
  v14 = *(_QWORD **)(v12 + 40);
  v15 = *(_QWORD *)(v13 + 40);
  if ( v14 != (_QWORD *)v15 )
  {
    LOBYTE(v16) = sub_15CC890(a6, (__int64)v14, *(_QWORD *)(v13 + 40));
    v17 = v16;
    if ( !(_BYTE)v16 )
    {
      if ( dword_4FBD1E0 > 4 )
        goto LABEL_4;
      v37 = *(_QWORD *)(v11 + 8);
      v38 = *(_QWORD *)(v46 + 8);
      v52[0] = v37;
      v51 = v38;
      if ( *((_DWORD *)sub_1C57390((__int64)(a1 + 4), &v51) + 2) <= 0x14u )
        return v17;
      v39 = a1 + 9;
      v40 = sub_1C55B00((__int64)(a1 + 8), (unsigned __int64 *)&v51);
      v41 = (__int64)(a1 + 8);
      if ( v40 == a1 + 9 )
        v45 = v51;
      else
        v45 = v40[5];
      v42 = sub_1C55B00(v41, v52);
      if ( v39 != v42 )
        v37 = v42[5];
      if ( v45 == v37 )
      {
LABEL_4:
        v18 = 0;
        while ( ++v18 <= (unsigned int)dword_4FBC220 )
        {
          v19 = *(unsigned int *)(a6 + 48);
          if ( !(_DWORD)v19 )
            goto LABEL_51;
          v20 = v19 - 1;
          v21 = *(_QWORD *)(a6 + 32);
          v22 = (v19 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = (_QWORD *)*v23;
          if ( v14 != (_QWORD *)*v23 )
          {
            v35 = 1;
            while ( v24 != (_QWORD *)-8LL )
            {
              v36 = v35 + 1;
              v22 = v20 & (v35 + v22);
              v23 = (__int64 *)(v21 + 16LL * v22);
              v24 = (_QWORD *)*v23;
              if ( v14 == (_QWORD *)*v23 )
                goto LABEL_11;
              v35 = v36;
            }
LABEL_51:
            BUG();
          }
LABEL_11:
          v25 = (__int64 *)(v21 + 16 * v19);
          if ( v25 == v23 )
            goto LABEL_51;
          v14 = *(_QWORD **)(v23[1] + 8);
          if ( v14 )
            v14 = (_QWORD *)*v14;
          v26 = v20 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v27 = (__int64 *)(v21 + 16LL * v26);
          v28 = *v27;
          if ( v15 != *v27 )
          {
            v33 = 1;
            while ( v28 != -8 )
            {
              v34 = v33 + 1;
              v26 = v20 & (v33 + v26);
              v27 = (__int64 *)(v21 + 16LL * v26);
              v28 = *v27;
              if ( v15 == *v27 )
                goto LABEL_15;
              v33 = v34;
            }
LABEL_50:
            BUG();
          }
LABEL_15:
          if ( v25 == v27 )
            goto LABEL_50;
          v29 = *(unsigned __int64 **)(v27[1] + 8);
          if ( !v29 )
            break;
          v15 = *v29;
          if ( *v29 == (_QWORD)v14 || *v29 == 0 || !v14 )
            break;
          if ( sub_15CC890(a6, (__int64)v14, v15) )
          {
            v43 = *(_QWORD *)(v46 + 24);
            if ( *(_BYTE *)(v43 + 16) <= 0x17u )
              v43 = 0;
            if ( (unsigned __int8)sub_1C61C30(a1, v43, (unsigned __int64)v14, a6) )
            {
              v44 = *(_QWORD *)(v47 + 24);
              if ( *(_BYTE *)(v44 + 16) <= 0x17u )
                v44 = 0;
              v17 = sub_1C61C30(a1, v44, v15, a6);
              if ( (_BYTE)v17 )
              {
                *a4 = v14;
                *a5 = v15;
                return v17;
              }
            }
            return 0;
          }
        }
      }
      return 0;
    }
    *a4 = v14;
    *a5 = v15;
    if ( a7 )
    {
      *a7 = 0;
      return v17;
    }
    return 1;
  }
  v31 = a2[2];
  if ( (unsigned int)v31 > 1 )
    v12 = *(_QWORD *)(v9[v31 - 1] + 16);
  LOBYTE(v32) = sub_1C612E0(a1, a6, v12, v13);
  v17 = v32;
  if ( (_BYTE)v32 )
  {
    *a4 = *(_QWORD *)(**(_QWORD **)a2 + 16LL);
    *a5 = *(_QWORD *)(v47 + 16);
    if ( a7 )
    {
      *a7 = 1;
      return v17;
    }
    return 1;
  }
  return 0;
}
