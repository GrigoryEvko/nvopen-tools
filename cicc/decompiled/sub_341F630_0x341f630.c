// Function: sub_341F630
// Address: 0x341f630
//
__int64 __fastcall sub_341F630(unsigned __int8 *a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // ecx
  unsigned __int64 v5; // rcx
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rdi
  int v9; // esi
  int v10; // r8d
  unsigned int v11; // edx
  unsigned __int8 *v12; // rcx
  __int64 v13; // rdx

  result = sub_B46490((__int64)a1);
  if ( (_BYTE)result )
    return 0;
  v4 = *a1;
  if ( (unsigned int)(v4 - 30) > 0xA )
  {
    if ( (_BYTE)v4 == 85 )
    {
      v13 = *((_QWORD *)a1 - 4);
      if ( !v13
        || *(_BYTE *)v13
        || *(_QWORD *)(v13 + 24) != *((_QWORD *)a1 + 10)
        || (*(_BYTE *)(v13 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v13 + 36) - 68) > 3 )
      {
LABEL_6:
        v7 = *(_DWORD *)(a2 + 144);
        v8 = *(_QWORD *)(a2 + 128);
        if ( v7 )
        {
          v9 = v7 - 1;
          v10 = 1;
          v11 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v12 = *(unsigned __int8 **)(v8 + 16LL * v11);
          if ( a1 == v12 )
            return result;
          while ( v12 != (unsigned __int8 *)-4096LL )
          {
            v11 = v9 & (v10 + v11);
            v12 = *(unsigned __int8 **)(v8 + 16LL * v11);
            if ( a1 == v12 )
              return result;
            ++v10;
          }
        }
        return 1;
      }
    }
    else
    {
      v5 = (unsigned int)(v4 - 39);
      if ( (unsigned int)v5 > 0x38 )
        goto LABEL_6;
      v6 = 0x100060000000001LL;
      if ( !_bittest64(&v6, v5) )
        goto LABEL_6;
    }
  }
  return result;
}
