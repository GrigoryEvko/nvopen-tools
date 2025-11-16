// Function: sub_38C6CD0
// Address: 0x38c6cd0
//
unsigned __int64 __fastcall sub_38C6CD0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // rcx
  unsigned __int64 v7; // rax
  _BYTE *v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned int v10; // ebx
  char v11; // si
  char v12; // r13
  unsigned __int32 v13; // eax
  __int16 v14; // ax
  _DWORD v15[11]; // [rsp-2Ch] [rbp-2Ch] BYREF

  result = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(unsigned int *)(v4 + 28);
  if ( result >= v5 )
  {
    v7 = result / v5;
    v8 = *(_BYTE **)(a3 + 24);
    v9 = *(_QWORD *)(a3 + 16);
    v10 = v7;
    if ( v7 <= 0x3F )
    {
      result = (unsigned int)v7 | 0x40;
      v11 = result;
      if ( v9 > (unsigned __int64)v8 )
      {
        *(_QWORD *)(a3 + 24) = v8 + 1;
        *v8 = result;
        return result;
      }
      return sub_16E7DE0(a3, v11);
    }
    if ( (v7 & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v12 = *(_BYTE *)(v4 + 16);
      if ( (v7 & 0xFFFFFFFFFFFF0000LL) != 0 )
      {
        if ( v9 <= (unsigned __int64)v8 )
        {
          sub_16E7DE0(a3, 4);
        }
        else
        {
          *(_QWORD *)(a3 + 24) = v8 + 1;
          *v8 = 4;
        }
        v13 = _byteswap_ulong(v10);
        if ( v12 != 0 || -(v12 == 0) == 1 )
          v13 = v10;
        v15[0] = v13;
        return sub_16E7EE0(a3, (char *)v15, 4u);
      }
      else
      {
        if ( v9 <= (unsigned __int64)v8 )
        {
          sub_16E7DE0(a3, 3);
        }
        else
        {
          *(_QWORD *)(a3 + 24) = v8 + 1;
          *v8 = 3;
        }
        v14 = __ROL2__(v10, 8);
        if ( v12 != 0 || -(v12 == 0) == 1 )
          v14 = v10;
        LOWORD(v15[0]) = v14;
        return sub_16E7EE0(a3, (char *)v15, 2u);
      }
    }
    else
    {
      if ( v9 <= (unsigned __int64)v8 )
      {
        sub_16E7DE0(a3, 2);
      }
      else
      {
        *(_QWORD *)(a3 + 24) = v8 + 1;
        *v8 = 2;
      }
      result = *(_QWORD *)(a3 + 24);
      if ( result >= *(_QWORD *)(a3 + 16) )
      {
        v11 = v10;
        return sub_16E7DE0(a3, v11);
      }
      *(_QWORD *)(a3 + 24) = result + 1;
      *(_BYTE *)result = v10;
    }
  }
  return result;
}
