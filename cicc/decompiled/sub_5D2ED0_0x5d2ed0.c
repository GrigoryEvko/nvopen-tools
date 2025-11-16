// Function: sub_5D2ED0
// Address: 0x5d2ed0
//
void __fastcall sub_5D2ED0(__int64 a1, __int64 a2, unsigned int a3)
{
  const char *v4; // rdi
  size_t v6; // rax
  char *v7; // r9
  size_t v8; // rbx
  size_t v9; // r15
  size_t v10; // rax
  _QWORD *v11; // r15
  unsigned int v12; // eax
  char *v13; // rbx
  char *v14; // rbx
  char v15; // al
  unsigned __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // esi
  __int64 v20; // r10
  char *src; // [rsp+0h] [rbp-110h]
  __int64 v22; // [rsp+8h] [rbp-108h]
  size_t v23; // [rsp+10h] [rbp-100h]
  size_t v24; // [rsp+18h] [rbp-F8h]
  char s[64]; // [rsp+20h] [rbp-F0h] BYREF
  char v26[64]; // [rsp+60h] [rbp-B0h] BYREF
  char dest[112]; // [rsp+A0h] [rbp-70h] BYREF

  v4 = *(const char **)(a1 + 8);
  if ( v4 && (*(_BYTE *)(a1 + 89) & 8) == 0 )
  {
    v24 = strlen(v4);
    if ( a2 && *(_QWORD *)(a2 + 8) )
    {
      v22 = *(_QWORD *)(a2 + 8);
      v6 = strlen((const char *)v22);
      strcpy(dest, "_ZZ");
      v23 = v6;
      if ( *(_BYTE *)v22 == 95 && *(_BYTE *)(v22 + 1) == 90 )
      {
        v23 = v6 - 2;
        v7 = (char *)(v22 + 2);
      }
      else
      {
        sprintf(s, "%lu", v6);
        strcpy(&dest[3], s);
        v7 = (char *)v22;
      }
    }
    else
    {
      strcpy(dest, "_ZZ");
      v7 = 0;
      v23 = 0;
    }
    src = v7;
    sprintf(s, "E%lu", v24);
    sprintf(v26, "_%lu", a3);
    v8 = strlen(dest);
    v9 = strlen(s);
    v10 = strlen(v26);
    v11 = (_QWORD *)sub_7247C0(v8 + v23 + v24 + 1 + v9 + v10);
    v12 = v8 + 1;
    if ( (unsigned int)(v8 + 1) >= 8 )
    {
      *v11 = *(_QWORD *)dest;
      *(_QWORD *)((char *)v11 + v12 - 8) = *(_QWORD *)&v26[v12 + 56];
      v16 = (unsigned __int64)(v11 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      v17 = ((_DWORD)v11 - v16 + v12) & 0xFFFFFFF8;
      if ( v17 >= 8 )
      {
        v18 = v17 & 0xFFFFFFF8;
        v19 = 0;
        do
        {
          v20 = v19;
          v19 += 8;
          *(_QWORD *)(v16 + v20) = *(_QWORD *)(dest - ((char *)v11 - v16) + v20);
        }
        while ( v19 < v18 );
      }
    }
    else if ( (v12 & 4) != 0 )
    {
      *(_DWORD *)v11 = *(_DWORD *)dest;
      *(_DWORD *)((char *)v11 + v12 - 4) = *(_DWORD *)&v26[v12 + 60];
    }
    else if ( (_DWORD)v8 != -1 )
    {
      *(_BYTE *)v11 = dest[0];
      if ( (v12 & 2) != 0 )
        *(_WORD *)((char *)v11 + v12 - 2) = *(_WORD *)&v26[v12 + 62];
    }
    v13 = (char *)v11 + v8;
    if ( src )
    {
      strcpy(v13, src);
      v13 += v23;
    }
    strcpy(v13, s);
    v14 = &v13[strlen(s)];
    strcpy(v14, *(const char **)(a1 + 8));
    strcpy(&v14[v24], v26);
    v15 = *(_BYTE *)(a1 + 89);
    *(_QWORD *)(a1 + 8) = v11;
    *(_BYTE *)(a1 + 89) = v15 & 0xF6 | 8;
  }
}
