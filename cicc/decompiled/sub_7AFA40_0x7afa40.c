// Function: sub_7AFA40
// Address: 0x7afa40
//
char *__fastcall sub_7AFA40(__int64 a1, __int64 a2)
{
  int v2; // edi
  char *v3; // r12
  __int64 *v4; // rbx
  unsigned int v5; // eax
  __int64 *v6; // rax
  char *result; // rax
  __int64 v8; // rax

  if ( qword_4F06438 )
    sub_7AF460((char *)1, a2);
  if ( unk_4D03CE4 )
  {
    v2 = byte_4F17F98;
  }
  else
  {
    byte_4F17F98 = 78;
    v2 = 78;
  }
  putc(v2, qword_4D04908);
  v3 = (char *)unk_4F06498;
  v4 = (__int64 *)unk_4F06458;
  if ( unk_4F06458 )
  {
    while ( 1 )
    {
      sub_7ABB60(v3, (char *)v4[1]);
      v5 = *((_DWORD *)v4 + 4);
      if ( v5 == 2 )
        break;
      if ( v5 > 2 )
      {
        if ( v5 != 3 )
          sub_721090();
        putc(32, qword_4D04908);
        v8 = v4[1];
        v4 = (__int64 *)*v4;
        v3 = (char *)(v8 + 2);
        goto LABEL_9;
      }
      if ( v5 )
      {
        v3 = (char *)v4[1];
        putc(92, qword_4D04908);
LABEL_8:
        putc(10, qword_4D04908);
        putc(byte_4F17F98, qword_4D04908);
        v4 = (__int64 *)*v4;
LABEL_9:
        if ( !v4 )
          goto LABEL_16;
      }
      else
      {
        fprintf(qword_4D04908, "??%c", (unsigned int)*((char *)v4 + 24));
        v6 = (__int64 *)*v4;
        v3 = (char *)v4[1];
        if ( *v4 && *((_DWORD *)v6 + 4) == 1 && (char *)v6[1] == v3 )
        {
          v4 = (__int64 *)*v4;
          goto LABEL_8;
        }
        v4 = (__int64 *)*v4;
        ++v3;
        if ( !v6 )
          goto LABEL_16;
      }
    }
    v3 = (char *)(v4[1] + 2);
    goto LABEL_8;
  }
LABEL_16:
  sub_7ABB60(v3, 0);
  result = sub_7AF460(0, 0);
  byte_4F17F98 = 0;
  return result;
}
