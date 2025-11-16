// Function: sub_1AD38C0
// Address: 0x1ad38c0
//
_QWORD *__fastcall sub_1AD38C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, char a6)
{
  _QWORD *v7; // r12
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rdx
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  _QWORD *v22; // rax
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  char v27; // al
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  _QWORD *v33; // [rsp+18h] [rbp-68h]
  _QWORD v34[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v35; // [rsp+30h] [rbp-50h] BYREF
  __int16 v36; // [rsp+40h] [rbp-40h]

  v7 = (_QWORD *)a1;
  v33 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
  v9 = sub_15F2060(a2);
  v10 = 36;
  v11 = sub_1632FA0(*(_QWORD *)(v9 + 40));
  v31 = a3 + 112;
  v12 = sub_1560180(v31, 36);
  v14 = a2;
  if ( !v12 )
  {
    v10 = 37;
    v27 = sub_1560180(v31, 37);
    v14 = a2;
    if ( !a6 && !v27 )
      goto LABEL_7;
  }
  if ( a5 > 1 )
  {
    v15 = *(_QWORD *)(a4 + 8);
    if ( v15 )
    {
      if ( !*(_QWORD *)(v15 + 16) )
        sub_4263D6(v31, v10, v13);
      v32 = v14;
      v16 = (*(__int64 (__fastcall **)(__int64, __int64))(v15 + 24))(v15, v9);
      v14 = v32;
      v15 = v16;
    }
    if ( (unsigned int)sub_1AE99B0(a1, a5, v11, v14, v15, 0) < a5 )
    {
LABEL_7:
      v17 = sub_15AAE50(v11, (__int64)v33);
      v18 = *(_DWORD *)(v11 + 4);
      if ( v17 >= a5 )
        a5 = v17;
      v34[0] = sub_1649960(a1);
      v35 = v34;
      v19 = *(_QWORD *)(v9 + 80);
      v34[1] = v20;
      v36 = 261;
      if ( !v19 )
        BUG();
      v21 = *(_QWORD *)(v19 + 24);
      if ( v21 )
        v21 -= 24;
      v22 = sub_1648A60(64, 1u);
      v7 = v22;
      if ( v22 )
        sub_15F8A50((__int64)v22, v33, v18, 0, a5, (__int64)&v35, v21);
      v25 = *(unsigned int *)(a4 + 48);
      if ( (unsigned int)v25 >= *(_DWORD *)(a4 + 52) )
      {
        sub_16CD150(a4 + 40, (const void *)(a4 + 56), 0, 8, v23, v24);
        v25 = *(unsigned int *)(a4 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a4 + 40) + 8 * v25) = v7;
      ++*(_DWORD *)(a4 + 48);
    }
  }
  return v7;
}
