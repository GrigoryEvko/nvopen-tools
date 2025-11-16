// Function: sub_31A5150
// Address: 0x31a5150
//
__int64 __fastcall sub_31A5150(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rdi
  char v8; // al
  unsigned __int8 *v9; // rsi
  unsigned int v10; // r12d
  char v11; // dl
  void *v13; // rax
  const void *v14; // rsi
  unsigned int v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]
  unsigned int v19; // [rsp+28h] [rbp-28h]

  v6 = a1[7];
  v18 = 0;
  v16 = 0;
  v17 = 0;
  v19 = 0;
  if ( v6 )
  {
    sub_C7D6A0(0, 0, 8);
    v7 = *(unsigned int *)(v6 + 144);
    v19 = v7;
    if ( (_DWORD)v7 )
    {
      v13 = (void *)sub_C7D670(16 * v7, 8);
      v14 = *(const void **)(v6 + 128);
      v17 = (__int64)v13;
      v18 = *(_QWORD *)(v6 + 136);
      memcpy(v13, v14, 16LL * v19);
    }
    else
    {
      v17 = 0;
      v18 = 0;
    }
  }
  v8 = sub_11F3070(**(_QWORD **)(*a1 + 32), a1[73], (__int64 *)a1[72]);
  v9 = a2;
  v10 = 0;
  v15 = sub_D34EB0(a1[2], v9, a3, *a1, (__int64)&v16, v8 ^ 1u, 0);
  if ( v11 && ((v15 + 1) & 0xFFFFFFFD) == 0 )
    v10 = v15;
  sub_C7D6A0(v17, 16LL * v19, 8);
  return v10;
}
