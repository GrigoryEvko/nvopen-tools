// Function: sub_3913FA0
// Address: 0x3913fa0
//
__int64 __fastcall sub_3913FA0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rbx
  int v5; // eax
  int v6; // r9d
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  int v12; // eax
  int v13; // r11d

  if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
    return sub_3914040();
  v4 = 0;
  v5 = *(_DWORD *)(a1 + 104);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = *(_QWORD *)(a1 + 88);
    v8 = *(_QWORD *)((*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    v9 = (v5 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v8 == *v10 )
    {
LABEL_5:
      v4 = v10[1];
    }
    else
    {
      v12 = 1;
      while ( v11 != -8 )
      {
        v13 = v12 + 1;
        v9 = v6 & (v12 + v9);
        v10 = (__int64 *)(v7 + 16LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
          goto LABEL_5;
        v12 = v13;
      }
      v4 = 0;
    }
  }
  return v4 + sub_38D0440(a3, a2);
}
