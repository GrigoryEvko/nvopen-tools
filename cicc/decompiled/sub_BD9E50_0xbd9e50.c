// Function: sub_BD9E50
// Address: 0xbd9e50
//
void __fastcall sub_BD9E50(char *src, char *a2)
{
  __int64 *v3; // r13
  unsigned __int64 v4; // rcx
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  unsigned __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rcx
  unsigned __int8 v10; // al
  __int64 *v11; // rdi

  if ( src != a2 )
  {
    v3 = (__int64 *)(src + 8);
    if ( a2 != src + 8 )
    {
      do
      {
        while ( 1 )
        {
          v8 = *v3;
          v9 = *(_QWORD *)(*(_QWORD *)(*v3 - 32LL * (*(_DWORD *)(*v3 + 4) & 0x7FFFFFF)) + 24LL);
          v10 = *(_BYTE *)(v9 - 16);
          v4 = (v10 & 2) != 0 ? *(_QWORD *)(v9 - 32) : -16 - 8LL * ((v10 >> 2) & 0xF) + v9;
          v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)src - 32LL * (*(_DWORD *)(*(_QWORD *)src + 4LL) & 0x7FFFFFF)) + 24LL);
          v6 = *(_BYTE *)(v5 - 16);
          v7 = (v6 & 2) != 0 ? *(_QWORD *)(v5 - 32) : -16 - 8LL * ((v6 >> 2) & 0xF) + v5;
          if ( v4 < v7 )
            break;
          v11 = v3++;
          sub_BD9DB0(v11);
          if ( a2 == (char *)v3 )
            return;
        }
        if ( src != (char *)v3 )
          memmove(src + 8, src, (char *)v3 - src);
        ++v3;
        *(_QWORD *)src = v8;
      }
      while ( a2 != (char *)v3 );
    }
  }
}
