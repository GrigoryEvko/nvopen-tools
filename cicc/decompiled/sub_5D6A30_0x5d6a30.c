// Function: sub_5D6A30
// Address: 0x5d6a30
//
void __fastcall sub_5D6A30(__int64 a1, _DWORD *a2)
{
  int v4; // r8d
  FILE *v5; // r13
  FILE *v6; // rdi
  FILE *v7; // r13
  FILE *v8; // rsi
  FILE *v9; // rdi
  int v10; // edi
  char *v11; // r14
  __int64 v12; // rsi
  FILE *v13; // rax
  FILE *v14; // rax
  char v15; // [rsp+7h] [rbp-29h] BYREF
  _BYTE v16[40]; // [rsp+8h] [rbp-28h] BYREF

  v4 = dword_4CF7CD4;
  a2[2] = 1;
  v5 = stream;
  if ( v4 )
  {
    if ( (*(_BYTE *)(a1 + 88) & 8) != 0 )
      goto LABEL_7;
  }
  else
  {
    v6 = qword_4CF7EA8;
    if ( !qword_4CF7EA8 )
    {
      v13 = (FILE *)sub_721330(0, a2);
      qword_4CF7F00 = 0;
      qword_4CF7EA8 = v13;
      v6 = v13;
      qword_4CF7F08 = 0;
      dword_4CF7F10 = 0;
    }
    sub_5D3B20(v6);
    if ( (*(_BYTE *)(a1 + 88) & 8) != 0 )
      goto LABEL_5;
  }
  if ( dword_4CF7EA0 )
  {
    sub_5D3EB0("#if 0");
    if ( dword_4CF7F40 )
      sub_5D37C0("#if 0", (unsigned int)dword_4CF7F40);
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
  }
LABEL_5:
  if ( v5 != stream )
    sub_5D3B20(v5);
LABEL_7:
  if ( *(_BYTE *)(a1 + 136) == 2 )
  {
    if ( (*(_BYTE *)(a1 + 88) & 0x70) != 0 )
      return;
    v7 = stream;
    v8 = stream;
    if ( !dword_4CF7CD4 )
    {
      v9 = qword_4CF7EA8;
      if ( !qword_4CF7EA8 )
      {
        v14 = (FILE *)sub_721330(0, stream);
        qword_4CF7F00 = 0;
        qword_4CF7EA8 = v14;
        v9 = v14;
        qword_4CF7F08 = 0;
        dword_4CF7F10 = 0;
      }
      sub_5D3B20(v9);
      v8 = stream;
    }
    v10 = 123;
    v11 = "static int __init_done=0; if (!__init_done) {__init_done=1;";
    while ( 1 )
    {
      putc(v10, v8);
      v10 = *v11++;
      if ( !(_BYTE)v10 )
        break;
      v8 = stream;
    }
    dword_4CF7F40 += 60;
    if ( v7 != stream )
      sub_5D3B20(v7);
    a2[4] = 1;
  }
  if ( !*a2 && *(_BYTE *)(a1 + 136) > 2u )
  {
    v12 = qword_4CF7E98;
    sub_72F9F0(a1, qword_4CF7E98, &v15, v16);
    if ( v15 == 3
      || v15 == 2 && (*(char *)(*(_QWORD *)(a1 + 184) + 50LL) < 0 || (unsigned int)sub_8D3B80(*(_QWORD *)(a1 + 120))) )
    {
      sub_5D68B0(a1, v12);
    }
  }
}
