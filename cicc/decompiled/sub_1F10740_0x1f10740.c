// Function: sub_1F10740
// Address: 0x1f10740
//
unsigned __int64 __fastcall sub_1F10740(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r10
  __int64 v7; // rax
  int v8; // edx
  int v9; // r11d

  result = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 368);
    v4 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      result = v3 + 16 * result;
      if ( v5 != (__int64 *)result )
      {
        v7 = v5[1];
        *v5 = -16;
        --*(_DWORD *)(a1 + 376);
        result = v7 & 0xFFFFFFFFFFFFFFF8LL;
        ++*(_DWORD *)(a1 + 380);
        *(_QWORD *)(result + 16) = 0;
      }
    }
    else
    {
      v8 = 1;
      while ( v6 != -8 )
      {
        v9 = v8 + 1;
        v4 = (result - 1) & (v8 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return result;
}
