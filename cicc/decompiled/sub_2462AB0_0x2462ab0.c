// Function: sub_2462AB0
// Address: 0x2462ab0
//
__int64 __fastcall sub_2462AB0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 *v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 48) == 33 )
  {
    v14 = *(_QWORD *)(a1 + 96);
    v10 = (__int64 *)sub_BCB120(*(_QWORD **)(a1 + 72));
    v15 = v14;
    v9 = 3;
    v16 = a5;
    v17 = a6;
  }
  else
  {
    v15 = a5;
    v9 = 2;
    v10 = *(__int64 **)(a1 + 496);
    v16 = a6;
  }
  v11 = sub_BCF480(v10, &v15, v9, 0);
  return sub_BA8C10(a2, a3, a4, v11, 0);
}
