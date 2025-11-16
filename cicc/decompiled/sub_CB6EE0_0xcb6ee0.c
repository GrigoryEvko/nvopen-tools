// Function: sub_CB6EE0
// Address: 0xcb6ee0
//
__int64 __fastcall sub_CB6EE0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 result; // rax
  __off_t v8; // rax
  int v9; // edi
  __off_t v10; // r12
  _OWORD v11[2]; // [rsp+0h] [rbp-60h] BYREF
  __int128 v12; // [rsp+20h] [rbp-40h]
  __int128 v13; // [rsp+30h] [rbp-30h]
  __int64 v14; // [rsp+40h] [rbp-20h]

  v5 = a4 ^ 1u;
  *(_DWORD *)(a1 + 8) = a5;
  *(_BYTE *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 44) = v5;
  *(_QWORD *)a1 = &unk_49DD190;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 48) = a2;
  *(_BYTE *)(a1 + 52) = a3;
  *(_WORD *)(a1 + 53) = 0;
  *(_BYTE *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  result = sub_2241E40(a1, a2, a3, v5, a5);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 80) = result;
  if ( (int)a2 < 0 )
  {
    *(_BYTE *)(a1 + 52) = 0;
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 1;
    if ( (int)a2 <= 2 )
      *(_BYTE *)(a1 + 52) = 0;
    v8 = lseek(a2, 0, 1);
    v9 = *(_DWORD *)(a1 + 48);
    v12 = 0;
    v10 = v8;
    v14 = 0;
    HIDWORD(v12) = 0xFFFF;
    memset(v11, 0, sizeof(v11));
    v13 = 0;
    result = sub_C82AC0(v9, (__int64)v11);
    *(_BYTE *)(a1 + 54) = DWORD2(v12) == 2;
    if ( (_DWORD)result || v10 == -1 )
    {
      *(_BYTE *)(a1 + 53) = 0;
      *(_QWORD *)(a1 + 88) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 88) = v10;
      *(_BYTE *)(a1 + 53) = 1;
    }
  }
  return result;
}
