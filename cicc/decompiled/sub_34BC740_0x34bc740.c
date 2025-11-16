// Function: sub_34BC740
// Address: 0x34bc740
//
__int64 __fastcall sub_34BC740(__int64 *a1)
{
  unsigned int v1; // r13d
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // rcx
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i si128; // [rsp+0h] [rbp-40h]
  char v12[48]; // [rsp+10h] [rbp-30h] BYREF

  v1 = (unsigned __int8)qword_503A6C8;
  if ( !(_BYTE)qword_503A6C8 )
    return v1;
  v3 = *a1;
  strcpy(v12, "mismatch");
  si128 = _mm_load_si128((const __m128i *)&xmmword_4386FA0);
  if ( (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
  {
    v4 = sub_B91C10(v3, 30);
    if ( v4 )
    {
      v5 = *(_BYTE *)(v4 - 16);
      if ( (v5 & 2) != 0 )
      {
        v7 = *(__int64 **)(v4 - 32);
        v6 = *(unsigned int *)(v4 - 24);
      }
      else
      {
        v6 = (*(_WORD *)(v4 - 16) >> 6) & 0xF;
        v7 = (__int64 *)(v4 - 16 - 8LL * ((v5 >> 2) & 0xF));
      }
      v8 = &v7[v6];
      if ( v8 != v7 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            if ( !*(_BYTE *)*v7 )
            {
              v9 = sub_B91420(*v7);
              if ( v10 == 24 && *(_OWORD *)&si128 == *(_OWORD *)v9 )
                break;
            }
            if ( v8 == ++v7 )
              return 0;
          }
          if ( *(_QWORD *)(v9 + 16) == *(_QWORD *)v12 )
            break;
          if ( v8 == ++v7 )
            return 0;
        }
        return v1;
      }
    }
  }
  return 0;
}
