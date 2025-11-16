// Function: sub_5F04F0
// Address: 0x5f04f0
//
__int64 __fastcall sub_5F04F0(__int64 a1, unsigned int a2, _DWORD *a3, int *a4)
{
  __int64 v5; // rbx
  char v6; // r14
  char v7; // al
  unsigned int v8; // r15d
  int v9; // r12d
  char *v10; // r8
  int v11; // r12d
  char v13; // al
  int v14; // eax
  __int64 v15; // rax
  char *v17; // [rsp+10h] [rbp-50h]
  int v18; // [rsp+24h] [rbp-3Ch] BYREF
  int v19; // [rsp+28h] [rbp-38h] BYREF
  char v20; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = a1;
  if ( a3 )
    *a3 = 0;
  if ( !a1 )
    goto LABEL_23;
  v6 = *(_BYTE *)(a1 + 80);
  v7 = v6;
  if ( v6 == 17 )
  {
    v5 = *(_QWORD *)(a1 + 88);
    if ( v5 )
    {
      v7 = *(_BYTE *)(v5 + 80);
      goto LABEL_5;
    }
LABEL_23:
    v11 = 1;
    v8 = 0;
    goto LABEL_8;
  }
LABEL_5:
  v19 = 0;
  v8 = 0;
  v9 = 0;
  v10 = &v20;
  if ( v7 == 10 )
    goto LABEL_11;
  while ( v6 == 17 )
  {
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
    v13 = *(_BYTE *)(v5 + 80);
    v19 = 0;
    if ( v13 == 10 )
    {
LABEL_11:
      v17 = v10;
      v14 = sub_5F04A0(v5, a2, (__int64)&v18, (__int64)&v19, (__int64)v10);
      v10 = v17;
      if ( v14 )
      {
        v15 = *(_QWORD *)(v5 + 88);
        if ( (*(_BYTE *)(v15 + 193) & 0x10) == 0 && (*(_BYTE *)(v15 + 206) & 0x18) == 0 && a3 )
          *a3 = 1;
        v8 = 1;
        if ( v18 )
        {
          if ( (v19 & 1) != 0 )
            v9 = 1;
        }
        else
        {
          v9 = 1;
        }
      }
    }
  }
  v11 = v8 ^ 1 | v9;
LABEL_8:
  *a4 = v11;
  return v8;
}
