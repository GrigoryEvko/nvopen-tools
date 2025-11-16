// Function: sub_ED3FC0
// Address: 0xed3fc0
//
unsigned __int64 *__fastcall sub_ED3FC0(unsigned __int64 *a1, __int64 a2, void *a3, size_t a4)
{
  int v6; // eax
  unsigned int v7; // r8d
  _QWORD *v8; // r9
  __int64 v10; // rax
  unsigned int v11; // r8d
  _QWORD *v12; // r9
  _QWORD *v13; // rcx
  _QWORD *v14; // [rsp+8h] [rbp-58h]
  _QWORD *v15; // [rsp+10h] [rbp-50h]
  unsigned int v16; // [rsp+1Ch] [rbp-44h]
  __int64 v17[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_ED3D70(v17, a2, a3, a4);
  if ( (v17[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v17[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v6 = sub_C92610();
  v7 = sub_C92740(a2 + 48, a3, a4, v6);
  v8 = (_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL * v7);
  if ( *v8 )
  {
    if ( *v8 != -8 )
    {
      *a1 = 1;
      return a1;
    }
    --*(_DWORD *)(a2 + 64);
  }
  v15 = v8;
  v16 = v7;
  v10 = sub_C7D670(a4 + 9, 8);
  v11 = v16;
  v12 = v15;
  v13 = (_QWORD *)v10;
  if ( a4 )
  {
    v14 = (_QWORD *)v10;
    memcpy((void *)(v10 + 8), a3, a4);
    v11 = v16;
    v12 = v15;
    v13 = v14;
  }
  *((_BYTE *)v13 + a4 + 8) = 0;
  *v13 = a4;
  *v12 = v13;
  ++*(_DWORD *)(a2 + 60);
  sub_C929D0((__int64 *)(a2 + 48), v11);
  *a1 = 1;
  return a1;
}
