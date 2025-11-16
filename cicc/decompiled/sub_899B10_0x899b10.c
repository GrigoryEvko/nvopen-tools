// Function: sub_899B10
// Address: 0x899b10
//
int __fastcall sub_899B10(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  char *v3; // rdi
  char v4; // al
  unsigned __int8 v5; // r12
  FILE *v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v10; // [rsp+8h] [rbp-58h] BYREF
  int v11; // [rsp+Ch] [rbp-54h] BYREF
  int v12; // [rsp+10h] [rbp-50h] BYREF
  int v13; // [rsp+14h] [rbp-4Ch] BYREF
  char *s; // [rsp+18h] [rbp-48h] BYREF
  char *v15; // [rsp+20h] [rbp-40h] BYREF
  FILE *stream; // [rsp+28h] [rbp-38h] BYREF
  __int64 v17; // [rsp+30h] [rbp-30h] BYREF
  __int64 v18[5]; // [rsp+38h] [rbp-28h] BYREF

  v1 = sub_729B10(*(_DWORD *)(a1 + 48), &v10, &v11, 0);
  if ( v1 )
  {
    v2 = v1;
    LODWORD(v1) = *(unsigned __int8 *)(v1 + 72);
    if ( (v1 & 0x80u) == 0LL )
    {
      v3 = *(char **)(v2 + 16);
      if ( v3 )
      {
        if ( (v1 & 1) == 0 )
        {
          v4 = v1 | 1;
          *(_BYTE *)(v2 + 72) = v4;
          v5 = (v4 & 8) != 0;
          LODWORD(v1) = sub_7B09E0(v3, 1, 1, v5, 0, 1, 0, 0, &s, &v15, (__int64 *)&stream, &v13, &v12, &v17);
          if ( (_DWORD)v1 )
          {
            if ( !sub_722E50(s, *(char **)(v2 + 8), 0, 0, 0) )
              goto LABEL_8;
            if ( !sub_722E50(s, qword_4F076F0, 0, 0, 0) )
              goto LABEL_8;
            LODWORD(v1) = sub_7AFD10(s, v18, 0, 1);
            if ( (_DWORD)v1 )
              goto LABEL_8;
            if ( v13 )
              return v1;
            if ( sub_7AFEF0(s, v18, 1, 1) )
            {
LABEL_8:
              LODWORD(v1) = v13;
              if ( !v13 )
                LODWORD(v1) = fclose(stream);
            }
            else
            {
              ++qword_4D03B78;
              v6 = stream;
              sub_7B1C00((__int64)stream, 0, (__int64)v15, s, 0, v5, 0, 0, 1, v12, v17, v18[0]);
              sub_666CF0((__int64)v6, 0, v7, v8);
              --qword_4D03B78;
              LODWORD(v1) = (unsigned int)sub_5F94C0(0);
              if ( dword_4F601E0 )
                LODWORD(v1) = sub_899AF0();
            }
          }
        }
      }
    }
  }
  return v1;
}
