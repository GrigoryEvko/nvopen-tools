// Function: sub_3990480
// Address: 0x3990480
//
void __fastcall sub_3990480(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _BYTE *v12; // rsi
  __int64 *v13; // [rsp-38h] [rbp-38h] BYREF
  _QWORD *v14; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a1 + 4508) != 1 )
  {
    v6 = a1 + 4040;
    if ( *(_BYTE *)(a1 + 4513) )
      v6 = a1 + 4520;
    v7 = sub_39A1860(v6 + 192, *(_QWORD *)(a1 + 8), a2, a3);
    v8 = *(_DWORD *)(a1 + 4508);
    if ( v8 == 2 )
    {
      v13 = (__int64 *)v7;
      v9 = (__int64 *)(a1 + 6352);
      v10 = *sub_398F1A0(
               a1 + 6456,
               (unsigned __int8 *)(v7 + 24),
               *(_QWORD *)v7,
               &v13,
               (__int64 (__fastcall **)(__int64, __int64))(a1 + 6488));
      v11 = (_QWORD *)sub_145CBF0(v9, 16, 16);
      v11[1] = a4;
      v14 = v11;
      *v11 = &unk_4A405B0;
      v12 = *(_BYTE **)(v10 + 32);
      if ( v12 == *(_BYTE **)(v10 + 40) )
      {
        sub_398F910(v10 + 24, v12, &v14);
      }
      else
      {
        if ( v12 )
        {
          *(_QWORD *)v12 = v11;
          v12 = *(_BYTE **)(v10 + 32);
        }
        *(_QWORD *)(v10 + 32) = v12 + 8;
      }
    }
    else if ( v8 == 3 )
    {
      sub_398FAA0(a1 + 5552, v7, a4);
    }
  }
}
