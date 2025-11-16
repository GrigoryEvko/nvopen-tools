// Function: sub_1594F50
// Address: 0x1594f50
//
__int64 __fastcall sub_1594F50(__int64 a1)
{
  __int64 **v1; // rsi
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // r8
  int v5; // ecx
  int v6; // ecx
  unsigned int v7; // r11d
  int v8; // ebx
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // r10
  __int64 ***v11; // r10
  unsigned int v12; // eax

  v1 = *(__int64 ***)(a1 - 48);
  v2 = *(_QWORD *)(a1 - 24);
  result = **v1;
  v4 = *(_QWORD *)result;
  v5 = *(_DWORD *)(*(_QWORD *)result + 1768LL);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = (unsigned int)v2 >> 9;
    v8 = 1;
    v9 = (((v7 ^ ((unsigned int)v2 >> 4) | ((unsigned __int64)(((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v7 ^ ((unsigned int)v2 >> 4)) << 32)) >> 22)
       ^ ((v7 ^ ((unsigned int)v2 >> 4) | ((unsigned __int64)(((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v7 ^ ((unsigned int)v2 >> 4)) << 32));
    v10 = ((9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13)))) >> 15)
        ^ (9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13))));
    for ( result = v6 & ((unsigned int)((v10 - 1 - (v10 << 27)) >> 31) ^ ((_DWORD)v10 - 1 - ((_DWORD)v10 << 27)));
          ;
          result = v6 & v12 )
    {
      v11 = (__int64 ***)(*(_QWORD *)(v4 + 1752) + 24LL * (unsigned int)result);
      if ( v1 == *v11 && (__int64 **)v2 == v11[1] )
        break;
      if ( *v11 == (__int64 **)-8LL && v11[1] == (__int64 **)-8LL )
        goto LABEL_9;
      v12 = v8 + result;
      ++v8;
    }
    *v11 = (__int64 **)-16LL;
    v11[1] = (__int64 **)-16LL;
    --*(_DWORD *)(v4 + 1760);
    ++*(_DWORD *)(v4 + 1764);
    v2 = *(_QWORD *)(a1 - 24);
LABEL_9:
    --*(_WORD *)(v2 + 18);
  }
  else
  {
    --*(_WORD *)(v2 + 18);
  }
  return result;
}
