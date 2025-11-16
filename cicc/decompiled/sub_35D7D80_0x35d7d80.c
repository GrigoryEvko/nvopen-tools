// Function: sub_35D7D80
// Address: 0x35d7d80
//
void __fastcall sub_35D7D80(char *src, char *a2)
{
  char *v3; // rbx
  __int64 v4; // r9
  __int64 v5; // r8
  int v6; // r15d
  unsigned int v7; // r13d
  bool v8; // cf
  bool v9; // al
  __int64 *v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // [rsp-48h] [rbp-48h]
  __int64 v13; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v3 = src + 24;
    if ( src + 24 != a2 )
    {
      while ( 1 )
      {
        v7 = *((_DWORD *)v3 + 5);
        v8 = *((_DWORD *)src + 5) < v7;
        if ( *((_DWORD *)src + 5) == v7 )
        {
          v11 = *((_DWORD *)src + 4);
          if ( *((_DWORD *)v3 + 4) != v11 )
          {
            v9 = *((_DWORD *)v3 + 4) > v11;
            goto LABEL_9;
          }
          v8 = *(_QWORD *)v3 < *(_QWORD *)src;
        }
        v9 = v8;
LABEL_9:
        if ( v9 )
        {
          v4 = *(_QWORD *)v3;
          v5 = *((_QWORD *)v3 + 1);
          v6 = *((_DWORD *)v3 + 4);
          if ( src != v3 )
          {
            v12 = *((_QWORD *)v3 + 1);
            v13 = *(_QWORD *)v3;
            memmove(src + 24, src, v3 - src);
            v5 = v12;
            v4 = v13;
          }
          v3 += 24;
          *(_QWORD *)src = v4;
          *((_QWORD *)src + 1) = v5;
          *((_DWORD *)src + 4) = v6;
          *((_DWORD *)src + 5) = v7;
          if ( a2 == v3 )
            return;
        }
        else
        {
          v10 = (__int64 *)v3;
          v3 += 24;
          sub_35D7D20(v10);
          if ( a2 == v3 )
            return;
        }
      }
    }
  }
}
