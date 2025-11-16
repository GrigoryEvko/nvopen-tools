// Function: sub_2AB4FA0
// Address: 0x2ab4fa0
//
unsigned __int64 __fastcall sub_2AB4FA0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  char v3; // dl
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 *v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // rbx
  __int64 v10; // rax
  bool v11; // of
  unsigned __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-58h]

  v3 = *a2;
  if ( *a2 == 61 )
    v4 = *((_QWORD *)a2 + 1);
  else
    v4 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
  if ( ((*(_BYTE *)(v4 + 8) - 7) & 0xFD) != 0 && (BYTE4(a3) || (_DWORD)a3 != 1) )
  {
    v5 = sub_BCE1B0((__int64 *)v4, a3);
    v3 = *a2;
    v4 = v5;
  }
  if ( v3 == 61 || (v6 = 0, v3 == 62) )
    v6 = *((_QWORD *)a2 - 4);
  v13 = v6;
  v7 = *(__int64 **)(a1 + 448);
  v8 = sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, (__int64)a2, *(unsigned int *)(a1 + 992), v6);
  v9 = sub_DFD550(v7, (unsigned int)*a2 - 29, v4, v13, v8);
  v10 = sub_DFDB90(*(_QWORD *)(a1 + 448));
  v11 = __OFADD__(v9, v10);
  result = v9 + v10;
  if ( v11 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v9 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
