// Function: sub_327FF20
// Address: 0x327ff20
//
__int64 __fastcall sub_327FF20(unsigned __int16 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned __int16 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  __int64 v10; // r12
  int v11; // edx
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // eax
  __int16 v15; // di
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-30h] BYREF
  char v19; // [rsp+8h] [rbp-28h]

  v4 = *a1;
  if ( *a1 )
  {
    if ( (unsigned __int16)(v4 - 17) > 0xD3u )
    {
      if ( v4 != 1 && (unsigned __int16)(v4 - 504) > 7u )
      {
        v5 = 16LL * (v4 - 1);
        v6 = *(_QWORD *)&byte_444C4A0[v5];
        v19 = byte_444C4A0[v5 + 8];
        v18 = v6;
        v7 = sub_CA1930(&v18);
        LOWORD(v8) = 2;
        if ( v7 != 1 )
        {
          LOWORD(v8) = 3;
          if ( v7 != 2 )
          {
            LOWORD(v8) = 4;
            if ( v7 != 4 )
            {
              LOWORD(v8) = 5;
              if ( v7 != 8 )
              {
                LOWORD(v8) = 6;
                if ( v7 != 16 )
                {
                  LOWORD(v8) = 7;
                  if ( v7 != 32 )
                  {
                    LOWORD(v8) = 8;
                    if ( v7 != 64 )
                      LOWORD(v8) = 9 * (v7 == 128);
                  }
                }
              }
            }
          }
        }
        goto LABEL_16;
      }
LABEL_32:
      BUG();
    }
    v10 = v4 - 1;
    v11 = (unsigned __int16)word_4456580[v10];
    if ( (unsigned __int16)v11 <= 1u || (unsigned __int16)(word_4456580[v10] - 504) <= 7u )
      goto LABEL_32;
    v12 = 16LL * (v11 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v12];
    v19 = byte_444C4A0[v12 + 8];
    v18 = v13;
    v14 = sub_CA1930(&v18);
    v15 = 2;
    if ( v14 != 1 )
    {
      v15 = 3;
      if ( v14 != 2 )
      {
        v15 = 4;
        if ( v14 != 4 )
        {
          v15 = 5;
          if ( v14 != 8 )
          {
            v15 = 6;
            if ( v14 != 16 )
            {
              v15 = 7;
              if ( v14 != 32 )
              {
                v15 = 8;
                if ( v14 != 64 )
                  v15 = 9 * (v14 == 128);
              }
            }
          }
        }
      }
    }
    v16 = word_4456340[v10];
    if ( (unsigned __int16)(v4 - 176) > 0x34u )
      LOWORD(v17) = sub_2D43050(v15, v16);
    else
      LOWORD(v17) = sub_2D43AD0(v15, v16);
  }
  else
  {
    if ( !sub_30070B0((__int64)a1) )
    {
      v8 = sub_30072B0((__int64)a1);
      v2 = v8;
LABEL_16:
      LOWORD(v2) = v8;
      return v2;
    }
    v17 = sub_300A990(a1, a2);
    v3 = v17;
  }
  LOWORD(v3) = v17;
  return v3;
}
