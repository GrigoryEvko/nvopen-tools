// Function: sub_15C9150
// Address: 0x15c9150
//
void __fastcall sub_15C9150(const char **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdx
  const char *v5; // rdi
  const char *v6; // rdx

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a2 )
  {
    v2 = a2;
    if ( *(_BYTE *)a2 == 15 )
    {
      v4 = a2;
    }
    else
    {
      v2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      if ( !v2 )
      {
        v6 = 0;
        v5 = byte_3F871B3;
LABEL_7:
        *a1 = v5;
        a1[1] = v6;
        a1[2] = (const char *)*(unsigned int *)(a2 + 28);
        return;
      }
      v4 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
    }
    v5 = *(const char **)(v4 - 8LL * *(unsigned int *)(v2 + 8));
    if ( v5 )
      v5 = (const char *)sub_161E970(v5);
    else
      v6 = 0;
    goto LABEL_7;
  }
}
