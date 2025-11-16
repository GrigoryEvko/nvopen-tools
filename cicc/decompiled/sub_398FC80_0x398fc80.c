// Function: sub_398FC80
// Address: 0x398fc80
//
void __fastcall sub_398FC80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v7; // rdi
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _BYTE *v12; // rsi
  __int64 *v13; // [rsp-38h] [rbp-38h] BYREF
  _QWORD *v14; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a1 + 4508) == 1 )
  {
    nullsub_2030();
  }
  else
  {
    v4 = a1 + 5752;
    v7 = a1 + 4040;
    if ( *(_BYTE *)(a1 + 4513) )
      v7 = a1 + 4520;
    v8 = sub_39A1860(v7 + 192, *(_QWORD *)(a1 + 8), a2, a3);
    v9 = *(_DWORD *)(a1 + 4508);
    if ( v9 == 2 )
    {
      v13 = (__int64 *)v8;
      v10 = *sub_398F1A0(
               v4 + 104,
               (unsigned __int8 *)(v8 + 24),
               *(_QWORD *)v8,
               &v13,
               (__int64 (__fastcall **)(__int64, __int64))(v4 + 136));
      v11 = (_QWORD *)sub_145CBF0((__int64 *)v4, 16, 16);
      v11[1] = a4;
      v14 = v11;
      *v11 = &unk_4A40580;
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
    else if ( v9 == 3 )
    {
      sub_398FAA0(a1 + 5552, v8, a4);
    }
  }
}
