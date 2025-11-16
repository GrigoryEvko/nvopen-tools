// Function: sub_30EC400
// Address: 0x30ec400
//
void __fastcall sub_30EC400(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r9
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // r11
  int v8; // eax
  int v9; // ebx

  v2 = *(unsigned int *)(a1 + 32);
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 16);
  if ( (_DWORD)v2 )
  {
    v5 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( v3 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v2) && v6[1] == a2 )
      {
        *v6 = -8192;
        --*(_DWORD *)(a1 + 24);
        ++*(_DWORD *)(a1 + 28);
      }
    }
    else
    {
      v8 = 1;
      while ( v7 != -4096 )
      {
        v9 = v8 + 1;
        v5 = (v2 - 1) & (v8 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( v3 == *v6 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
}
