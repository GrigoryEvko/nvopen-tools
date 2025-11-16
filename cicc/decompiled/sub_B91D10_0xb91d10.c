// Function: sub_B91D10
// Address: 0xb91d10
//
void __fastcall sub_B91D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned int v6; // ecx
  __int64 v7; // r8
  unsigned int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // r9
  int v11; // edi
  int v12; // r11d

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v5 = *(_QWORD *)sub_BD5C60(a1, a2);
    v6 = *(_DWORD *)(v5 + 3248);
    v7 = *(_QWORD *)(v5 + 3232);
    if ( v6 )
    {
      v8 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v9 = (__int64 *)(v7 + 40LL * v8);
      v10 = *v9;
      if ( a1 == *v9 )
      {
LABEL_4:
        sub_B91B70(v9 + 1, a2, a3);
        return;
      }
      v11 = 1;
      while ( v10 != -4096 )
      {
        v12 = v11 + 1;
        v8 = (v6 - 1) & (v11 + v8);
        v9 = (__int64 *)(v7 + 40LL * v8);
        v10 = *v9;
        if ( a1 == *v9 )
          goto LABEL_4;
        v11 = v12;
      }
    }
    v9 = (__int64 *)(v7 + 40LL * v6);
    goto LABEL_4;
  }
}
