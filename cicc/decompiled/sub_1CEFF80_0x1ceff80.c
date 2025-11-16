// Function: sub_1CEFF80
// Address: 0x1ceff80
//
void __fastcall sub_1CEFF80(_QWORD *src, _QWORD *a2)
{
  _DWORD *v2; // r8
  unsigned int v3; // r13d
  __int64 v4; // r9
  _DWORD *v5; // r15
  __int64 v6; // r14
  unsigned int v7; // eax
  _DWORD *v8; // r15
  __int64 v9; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v2 = src + 3;
    if ( a2 != src + 3 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *((_QWORD *)v2 + 1);
          v7 = *(_DWORD *)(src[1] + 48LL);
          if ( *(_DWORD *)(v6 + 48) != v7 )
            break;
          v3 = v2[4];
          if ( v3 >= *((_DWORD *)src + 4) )
            goto LABEL_10;
LABEL_5:
          v4 = *(_QWORD *)v2;
          v5 = v2 + 6;
          if ( src != (_QWORD *)v2 )
          {
            v9 = *(_QWORD *)v2;
            memmove(src + 3, src, (char *)v2 - (char *)src);
            v4 = v9;
          }
          *src = v4;
          v2 = v5;
          src[1] = v6;
          *((_DWORD *)src + 4) = v3;
          if ( a2 == (_QWORD *)v5 )
            return;
        }
        if ( *(_DWORD *)(v6 + 48) < v7 )
        {
          v3 = v2[4];
          goto LABEL_5;
        }
LABEL_10:
        v8 = v2 + 6;
        sub_1CEFF30((__int64 *)v2);
        v2 = v8;
      }
      while ( a2 != (_QWORD *)v8 );
    }
  }
}
