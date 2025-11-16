// Function: sub_1C105D0
// Address: 0x1c105d0
//
__int64 __fastcall sub_1C105D0(size_t a1, const char *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 result; // rax
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rdi
  void *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rdi
  _BYTE *v16; // rax
  _DWORD v17[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = (__int64)a2;
  if ( (unsigned __int8)sub_1C07F10(a1, (__int64)a2, v17, a3) )
  {
    result = v17[0];
  }
  else
  {
    result = sub_1C10020(a1, (__int64)a2, a3);
    v17[0] = result;
  }
  if ( byte_4FBA340 && !(_DWORD)result )
  {
    v6 = sub_16E8CB0();
    v9 = v6[3];
    v10 = (__int64)v6;
    if ( (unsigned __int64)(v6[2] - v9) <= 6 )
    {
      a2 = "Invar: ";
      sub_16E7EE0((__int64)v6, "Invar: ", 7u);
    }
    else
    {
      *(_DWORD *)v9 = 1635151433;
      *(_WORD *)(v9 + 4) = 14962;
      *(_BYTE *)(v9 + 6) = 32;
      v6[3] += 7LL;
    }
    v11 = sub_16E8C20(v10, (__int64)a2, v9, v7, v8);
    sub_155C2B0(v4, (__int64)v11, 0);
    v15 = sub_16E8C20(v4, (__int64)v11, v12, v13, v14);
    v16 = (_BYTE *)v15[3];
    if ( (_BYTE *)v15[2] == v16 )
    {
      sub_16E7EE0((__int64)v15, "\n", 1u);
    }
    else
    {
      *v16 = 10;
      ++v15[3];
    }
    return v17[0];
  }
  return result;
}
