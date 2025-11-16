// Function: sub_10A0880
// Address: 0x10a0880
//
_QWORD *__fastcall sub_10A0880(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  _QWORD *v6; // r12
  _QWORD **v8; // rdx
  int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE v17[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v18; // [rsp+30h] [rbp-40h]

  v5 = sub_AD6530(*(_QWORD *)(a2 + 8), a2);
  v6 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 56LL))(
                   a1[10],
                   40,
                   a2,
                   v5);
  if ( !v6 )
  {
    v18 = 257;
    v6 = sub_BD2C40(72, unk_3F10FD0);
    if ( v6 )
    {
      v8 = *(_QWORD ***)(a2 + 8);
      v9 = *((unsigned __int8 *)v8 + 8);
      if ( (unsigned int)(v9 - 17) > 1 )
      {
        v11 = sub_BCB2A0(*v8);
      }
      else
      {
        BYTE4(v16) = (_BYTE)v9 == 18;
        LODWORD(v16) = *((_DWORD *)v8 + 8);
        v10 = (__int64 *)sub_BCB2A0(*v8);
        v11 = sub_BCE1B0(v10, v16);
      }
      sub_B523C0((__int64)v6, v11, 53, 40, a2, v5, (__int64)v17, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v6,
      a3,
      a1[7],
      a1[8]);
    v12 = *a1;
    v13 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v13 )
    {
      do
      {
        v14 = *(_QWORD *)(v12 + 8);
        v15 = *(_DWORD *)v12;
        v12 += 16;
        sub_B99FD0((__int64)v6, v15, v14);
      }
      while ( v13 != v12 );
    }
  }
  return v6;
}
