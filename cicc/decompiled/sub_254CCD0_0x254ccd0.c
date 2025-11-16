// Function: sub_254CCD0
// Address: 0x254ccd0
//
__int64 __fastcall sub_254CCD0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 result; // rax
  int v7; // ecx
  __int64 v8; // r8
  __int64 v9; // rsi
  int v10; // ecx
  int v11; // r12d
  __int64 v12; // rdi
  _QWORD *v13; // rdx
  unsigned int v14; // eax

  v3 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  if ( sub_B2FC80((__int64)v3) || (result = sub_B2FC00(v3), (_BYTE)result) )
  {
    result = sub_B19060(*(_QWORD *)(a2 + 208) + 248LL, (__int64)v3, v4, v5);
    if ( !(_BYTE)result )
    {
      if ( !*(_QWORD *)(a2 + 4432)
        || (result = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(a2 + 4440))(a2 + 4416, v3), !(_BYTE)result) )
      {
        result = *(unsigned __int8 *)(a1 + 104);
        *(_BYTE *)(a1 + 105) = result;
      }
    }
  }
  v7 = *(_DWORD *)(a2 + 56);
  v8 = *(_QWORD *)(a2 + 40);
  if ( v7 )
  {
    v9 = *(_QWORD *)(a1 + 72);
    v10 = v7 - 1;
    v11 = 1;
    v12 = *(_QWORD *)(a1 + 80);
    for ( result = v10
                 & (((unsigned int)v12 >> 9)
                  ^ ((unsigned int)*(_QWORD *)(a1 + 80) >> 4)
                  ^ (16 * (((unsigned int)v9 >> 9) ^ ((unsigned int)*(_QWORD *)(a1 + 72) >> 4)))); ; result = v10 & v14 )
    {
      v13 = (_QWORD *)(v8 + ((unsigned __int64)(unsigned int)result << 6));
      if ( v9 == *v13 && v12 == v13[1] )
        break;
      if ( unk_4FEE4D0 == *v13 && unk_4FEE4D8 == v13[1] )
        return result;
      v14 = v11 + result;
      ++v11;
    }
    result = *(unsigned __int8 *)(a1 + 104);
    *(_BYTE *)(a1 + 105) = result;
  }
  return result;
}
