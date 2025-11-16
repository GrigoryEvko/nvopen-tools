// Function: sub_103B590
// Address: 0x103b590
//
__int64 __fastcall sub_103B590(__int64 a1, __int64 *a2, _DWORD *a3, unsigned int a4)
{
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  unsigned __int64 v10; // rdx
  _BYTE *v11; // rsi
  __int64 result; // rax
  __int64 v13; // rsi
  __int64 v14[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v16[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = (unsigned int)*a3;
  v7 = a2[1];
  v8 = a4 - (unsigned int)v6;
  if ( v6 > v7 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
  v10 = v7 - v6;
  v11 = (_BYTE *)(v6 + *a2);
  v15[0] = (__int64)v16;
  if ( v10 > v8 )
    v10 = v8;
  sub_103ABA0(v15, v11, (__int64)&v11[v10]);
  v14[0] = v15[0];
  v14[1] = v15[1];
  result = sub_C93AD0(v14, " = MemoryDef(", 0xDu);
  if ( !result )
  {
    result = sub_C93AD0(v14, " = MemoryPhi(", 0xDu);
    if ( !result )
    {
      result = sub_C93AD0(v14, "MemoryUse(", 0xAu);
      if ( !result )
      {
        result = *a2;
        v13 = (unsigned int)*a3;
        if ( *a2 + a4 == *a2 + a2[1] )
        {
          a2[1] = v13;
          *(_BYTE *)(result + v13) = 0;
        }
        else
        {
          result = sub_2240CE0(a2, v13, a4 - v13);
        }
        --*a3;
      }
    }
  }
  if ( (_QWORD *)v15[0] != v16 )
    return j_j___libc_free_0(v15[0], v16[0] + 1LL);
  return result;
}
