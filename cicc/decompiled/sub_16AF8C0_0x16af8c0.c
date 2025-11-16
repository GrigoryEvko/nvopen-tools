// Function: sub_16AF8C0
// Address: 0x16af8c0
//
__int64 __fastcall sub_16AF8C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  size_t v4; // rax
  unsigned __int64 v5; // rcx
  size_t v6; // rdx
  _QWORD *v7; // rax
  _BYTE *v9; // rsi
  unsigned __int64 v10; // rdi
  _BYTE *v11; // rsi
  char *v12; // r14
  unsigned int v13; // esi
  unsigned int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rcx
  unsigned __int64 v17; // r8
  char s[32]; // [rsp+0h] [rbp-B0h] BYREF
  time_t timer[4]; // [rsp+20h] [rbp-90h] BYREF
  struct tm tp; // [rsp+40h] [rbp-70h] BYREF

  v2 = a1;
  timer[0] = a2 / 1000000000;
  localtime_r(timer, &tp);
  strftime(s, 0x14u, "%Y-%m-%d %H:%M:%S", &tp);
  v4 = strlen(s);
  v5 = *(_QWORD *)(a1 + 16);
  v6 = v4;
  v7 = *(_QWORD **)(a1 + 24);
  if ( v6 <= v5 - (unsigned __int64)v7 )
  {
    if ( !v6 )
      goto LABEL_3;
    v9 = *(_BYTE **)(a1 + 24);
    if ( (unsigned int)v6 >= 8 )
    {
      v10 = (unsigned __int64)(v7 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *v7 = *(_QWORD *)s;
      *(_QWORD *)&v9[(unsigned int)v6 - 8] = *(_QWORD *)&s[(unsigned int)v6 - 8];
      v11 = &v9[-v10];
      v12 = (char *)(s - v11);
      v13 = (v6 + (_DWORD)v11) & 0xFFFFFFF8;
      if ( v13 >= 8 )
      {
        v14 = v13 & 0xFFFFFFF8;
        v15 = 0;
        do
        {
          v16 = v15;
          v15 += 8;
          *(_QWORD *)(v10 + v16) = *(_QWORD *)&v12[v16];
        }
        while ( v15 < v14 );
      }
      goto LABEL_10;
    }
    if ( (v6 & 4) != 0 )
    {
      *(_DWORD *)v7 = *(_DWORD *)s;
      *(_DWORD *)&v9[(unsigned int)v6 - 4] = *(_DWORD *)&s[(unsigned int)v6 - 4];
      v17 = *(_QWORD *)(a1 + 16);
      v9 = *(_BYTE **)(a1 + 24);
    }
    else
    {
      v17 = *(_QWORD *)(a1 + 16);
      if ( (_DWORD)v6 )
      {
        *v9 = s[0];
        if ( (v6 & 2) == 0 )
        {
LABEL_10:
          v17 = *(_QWORD *)(v2 + 16);
          v9 = *(_BYTE **)(v2 + 24);
          goto LABEL_11;
        }
        *(_WORD *)&v9[(unsigned int)v6 - 2] = *(_WORD *)&s[(unsigned int)v6 - 2];
        v17 = *(_QWORD *)(a1 + 16);
        v9 = *(_BYTE **)(a1 + 24);
      }
    }
LABEL_11:
    v7 = &v9[v6];
    *(_QWORD *)(v2 + 24) = &v9[v6];
    if ( v17 > (unsigned __int64)&v9[v6] )
      goto LABEL_4;
LABEL_12:
    v2 = sub_16E7DE0(v2, 46);
    goto LABEL_5;
  }
  v2 = sub_16E7EE0(a1, s);
  v7 = *(_QWORD **)(v2 + 24);
  v5 = *(_QWORD *)(v2 + 16);
LABEL_3:
  if ( v5 <= (unsigned __int64)v7 )
    goto LABEL_12;
LABEL_4:
  *(_QWORD *)(v2 + 24) = (char *)v7 + 1;
  *(_BYTE *)v7 = 46;
LABEL_5:
  timer[1] = (time_t)"%.9lu";
  timer[0] = (time_t)&unk_49EEAD0;
  timer[2] = a2 % 1000000000;
  return sub_16E8450(v2, timer);
}
