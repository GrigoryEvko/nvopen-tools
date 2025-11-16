// Function: sub_2334520
// Address: 0x2334520
//
__int64 __fastcall sub_2334520(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  int v15; // edx
  char v16; // al
  __int64 v17; // [rsp+0h] [rbp-C0h] BYREF
  unsigned __int64 v18; // [rsp+8h] [rbp-B8h]
  __int64 v19; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v20; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-98h]
  unsigned __int64 v22[4]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v23[4]; // [rsp+50h] [rbp-70h] BYREF
  char v24; // [rsp+70h] [rbp-50h]
  _QWORD v25[2]; // [rsp+78h] [rbp-48h] BYREF
  _QWORD *v26; // [rsp+88h] [rbp-38h] BYREF

  v17 = a2;
  v18 = a3;
  if ( a3 )
  {
    while ( 1 )
    {
      v20 = 0;
      v21 = 0;
      LOBYTE(v23[0]) = 59;
      v4 = sub_C931B0(&v17, v23, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = v17;
        v4 = v18;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = v17;
        if ( v4 + 1 > v18 )
        {
          v5 = v18;
          v7 = 0;
        }
        else
        {
          v7 = v18 - v5;
        }
        v8 = v17 + v5;
        if ( v4 > v18 )
          v4 = v18;
      }
      v20 = v6;
      v21 = v4;
      v17 = v8;
      v18 = v7;
      if ( v4 == 3 )
      {
        if ( *(_WORD *)v6 != 24941 || *(_BYTE *)(v6 + 2) != 121 )
          goto LABEL_9;
        v15 = 0;
      }
      else
      {
        if ( v4 != 4 || *(_DWORD *)v6 != 1953723757 )
        {
LABEL_9:
          v9 = sub_C63BB0();
          v23[1] = 38;
          v10 = v9;
          v12 = v11;
          v23[0] = "invalid StackLifetime parameter '{0}' ";
          v23[2] = &v26;
          v23[3] = 1;
          v24 = 1;
          v25[0] = &unk_49DB108;
          v25[1] = &v20;
          v26 = v25;
          sub_23328D0((__int64)v22, (__int64)v23);
          sub_23058C0(&v19, (__int64)v22, v10, v12);
          v13 = v19;
          *(_BYTE *)(a1 + 8) |= 3u;
          *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
          sub_2240A30(v22);
          return a1;
        }
        v15 = 1;
      }
      if ( !v7 )
        goto LABEL_15;
    }
  }
  v15 = 0;
LABEL_15:
  v16 = *(_BYTE *)(a1 + 8);
  *(_DWORD *)a1 = v15;
  *(_BYTE *)(a1 + 8) = v16 & 0xFC | 2;
  return a1;
}
