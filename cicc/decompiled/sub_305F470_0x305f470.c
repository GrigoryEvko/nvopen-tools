// Function: sub_305F470
// Address: 0x305f470
//
__int64 __fastcall sub_305F470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v7; // [rsp+Ch] [rbp-14h]

  v3 = (__int64 *)a3;
  if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
    v3 = **(__int64 ***)(a3 + 16);
  v4 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v3, 0);
  BYTE2(v7) = 0;
  return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 32) + 736LL))(
           *(_QWORD *)(a1 + 32),
           *v3,
           v4,
           v5,
           v7);
}
