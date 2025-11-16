// Function: sub_24647C0
// Address: 0x24647c0
//
unsigned __int64 __fastcall sub_24647C0(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  _BYTE *v6; // r14
  __int64 *v7; // rdi
  __int64 **v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // rax
  int v11; // [rsp+8h] [rbp-68h]
  _QWORD v12[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v13; // [rsp+30h] [rbp-40h]

  result = 0;
  v5 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v5 + 4) )
  {
    v13 = 257;
    v6 = sub_94BCF0((unsigned int **)a2, *(_QWORD *)(v5 + 112), *(_QWORD *)(v5 + 80), (__int64)v12);
    if ( a3 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      v13 = 257;
      v10 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v9 + 80), a3, 0);
      v6 = (_BYTE *)sub_929C50((unsigned int **)a2, v6, v10, (__int64)v12, 0, 0);
    }
    v7 = *(__int64 **)(a2 + 72);
    v12[0] = "_msarg_o";
    v13 = 259;
    v8 = (__int64 **)sub_BCE3C0(v7, 0);
    return sub_24633A0((__int64 *)a2, 0x30u, (unsigned __int64)v6, v8, (__int64)v12, 0, v11, 0);
  }
  return result;
}
