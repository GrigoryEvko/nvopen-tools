// Function: sub_2CBA1F0
// Address: 0x2cba1f0
//
__int64 __fastcall sub_2CBA1F0(__int64 a1, __int64 a2, _QWORD *a3, __int64 *a4)
{
  unsigned int v4; // r15d
  _QWORD *v6; // rcx
  __int64 v8; // r12
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r12
  const char *v13; // r15
  const char *v14; // rax
  size_t v15; // rdx
  bool v16; // cc
  size_t v17; // r8
  size_t v18; // rdx
  unsigned int v19; // eax
  size_t v20; // [rsp+0h] [rbp-40h]
  _QWORD *v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  LOBYTE(v4) = 1;
  v6 = (_QWORD *)(a1 + 8);
  if ( !a2 && a3 != v6 )
  {
    v22 = *a4;
    v10 = sub_BD5D20(a3[4]);
    v12 = v11;
    v13 = v10;
    v14 = sub_BD5D20(v22);
    v6 = (_QWORD *)(a1 + 8);
    v16 = v15 <= v12;
    v17 = v15;
    v18 = v12;
    if ( v16 )
      v18 = v17;
    if ( v18 && (v20 = v17, v19 = memcmp(v14, v13, v18), v6 = (_QWORD *)(a1 + 8), v17 = v20, v19) )
    {
      v4 = v19 >> 31;
    }
    else
    {
      LOBYTE(v4) = v17 < v12;
      if ( v17 == v12 )
        LOBYTE(v4) = 0;
    }
  }
  v21 = v6;
  v8 = sub_22077B0(0x28u);
  *(_QWORD *)(v8 + 32) = *a4;
  sub_220F040(v4, v8, a3, v21);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
