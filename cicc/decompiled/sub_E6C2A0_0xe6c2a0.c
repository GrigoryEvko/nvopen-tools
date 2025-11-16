// Function: sub_E6C2A0
// Address: 0xe6c2a0
//
__int64 __fastcall sub_E6C2A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  char v8; // al
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  __int64 v11; // [rsp+8h] [rbp-28h]
  __int64 *v12; // [rsp+10h] [rbp-20h]
  __int64 v13; // [rsp+18h] [rbp-18h]
  __int16 v14; // [rsp+20h] [rbp-10h]

  v5 = *(_QWORD **)(a1 + 152);
  v6 = v5[16];
  if ( v6 )
  {
    v7 = v5[15];
  }
  else
  {
    v7 = v5[11];
    v6 = v5[12];
  }
  v8 = *((_BYTE *)a2 + 32);
  if ( v8 )
  {
    if ( v8 == 1 )
    {
      v10 = v7;
      v11 = v6;
      v14 = 261;
    }
    else
    {
      if ( *((_BYTE *)a2 + 33) == 1 )
      {
        a5 = a2[1];
        a2 = (__int64 *)*a2;
      }
      else
      {
        v8 = 2;
      }
      v10 = v7;
      v11 = v6;
      v12 = a2;
      v13 = a5;
      LOBYTE(v14) = 5;
      HIBYTE(v14) = v8;
    }
  }
  else
  {
    v14 = 256;
  }
  return sub_E6BFC0((_DWORD *)a1, (__int64)&v10, 1, 0);
}
