// Function: sub_30D1EE0
// Address: 0x30d1ee0
//
void __fastcall sub_30D1EE0(__int64 a1, __int64 a2)
{
  unsigned int v2; // esi
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // r9
  __int64 *v7; // rdi
  int v8; // r11d
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r8
  _DWORD *v12; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v16; // [rsp+18h] [rbp-28h] BYREF

  v15 = a2;
  if ( (_BYTE)qword_5030CC8 )
  {
    v2 = *(_DWORD *)(a1 + 696);
    v4 = a1 + 672;
    if ( v2 )
    {
      v5 = v15;
      v6 = *(_QWORD *)(a1 + 680);
      v7 = 0;
      v8 = 1;
      v9 = (v2 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v10 = (_QWORD *)(v6 + 24LL * v9);
      v11 = *v10;
      if ( v15 == *v10 )
      {
LABEL_4:
        v12 = v10 + 1;
LABEL_5:
        *v12 = *(_DWORD *)(a1 + 716);
        v12[2] = *(_DWORD *)(a1 + 704);
        return;
      }
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v7 )
          v7 = v10;
        v9 = (v2 - 1) & (v8 + v9);
        v10 = (_QWORD *)(v6 + 24LL * v9);
        v11 = *v10;
        if ( v15 == *v10 )
          goto LABEL_4;
        ++v8;
      }
      if ( !v7 )
        v7 = v10;
      v13 = *(_DWORD *)(a1 + 688);
      ++*(_QWORD *)(a1 + 672);
      v14 = v13 + 1;
      v16 = v7;
      if ( 4 * (v13 + 1) < 3 * v2 )
      {
        if ( v2 - *(_DWORD *)(a1 + 692) - v14 > v2 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a1 + 688) = v14;
          if ( *v7 != -4096 )
            --*(_DWORD *)(a1 + 692);
          *v7 = v5;
          v12 = v7 + 1;
          v7[1] = 0;
          v7[2] = 0;
          goto LABEL_5;
        }
LABEL_21:
        sub_30D1CF0(v4, v2);
        sub_30D1010(v4, &v15, &v16);
        v5 = v15;
        v7 = v16;
        v14 = *(_DWORD *)(a1 + 688) + 1;
        goto LABEL_16;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 672);
      v16 = 0;
    }
    v2 *= 2;
    goto LABEL_21;
  }
}
