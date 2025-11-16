// Function: sub_2FE3F70
// Address: 0x2fe3f70
//
__int64 __fastcall sub_2FE3F70(__int64 a1, __int64 *a2, _BYTE *a3, int a4)
{
  __int16 v4; // r14
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  unsigned int v10; // esi
  _BYTE v12[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  if ( !byte_3F8E4E0[8 * a4 + 5] )
    return 0;
  v4 = a4;
  v5 = 0;
  if ( (unsigned __int8)sub_B46540(a3) )
  {
    v13 = 257;
    v6 = sub_BD2C40(80, unk_3F222C8);
    v5 = (__int64)v6;
    if ( v6 )
      sub_B4D930((__int64)v6, a2[9], v4, 1, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v5,
      v12,
      a2[7],
      a2[8]);
    v7 = *a2;
    v8 = *a2 + 16LL * *((unsigned int *)a2 + 2);
    if ( *a2 != v8 )
    {
      do
      {
        v9 = *(_QWORD *)(v7 + 8);
        v10 = *(_DWORD *)v7;
        v7 += 16;
        sub_B99FD0(v5, v10, v9);
      }
      while ( v8 != v7 );
    }
  }
  return v5;
}
