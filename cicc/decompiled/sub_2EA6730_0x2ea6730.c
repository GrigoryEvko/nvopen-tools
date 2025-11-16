// Function: sub_2EA6730
// Address: 0x2ea6730
//
__int64 __fastcall sub_2EA6730(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int8 v4; // al
  __int64 *v5; // rax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  int v11; // ebx
  unsigned int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v1 = sub_2EA6550(a1);
  v15 = v1;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 16);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 != v2 + 48 )
      {
        if ( !v3 )
LABEL_15:
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 <= 0xA && (*(_BYTE *)(v3 - 17) & 0x20) != 0 )
        {
          v15 = sub_B91C10(v3 - 24, 18);
LABEL_8:
          if ( v15 )
          {
            v4 = *(_BYTE *)(v15 - 16);
            if ( (v4 & 2) != 0 )
            {
              if ( *(_DWORD *)(v15 - 24) )
              {
                v5 = *(__int64 **)(v15 - 32);
                goto LABEL_12;
              }
            }
            else if ( (*(_WORD *)(v15 - 16) & 0x3C0) != 0 )
            {
              v5 = (__int64 *)(v15 - 8LL * ((v4 >> 2) & 0xF) - 16);
LABEL_12:
              result = *v5;
              if ( v15 == result )
                return result;
            }
          }
        }
      }
    }
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 32);
    v16 = v7;
    if ( *(_QWORD *)v7 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
      if ( v8 )
      {
        v14 = *(_QWORD *)(a1 + 40);
        if ( v7 != v14 )
        {
          v9 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
          do
          {
            v10 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v10 == v9 + 48 )
              break;
            if ( !v10 )
              goto LABEL_15;
            if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
              break;
            v11 = sub_B46E30(v10 - 24);
            if ( v11 )
            {
              v12 = 0;
              while ( v8 != sub_B46EC0(v10 - 24, v12) )
              {
                if ( v11 == ++v12 )
                  goto LABEL_31;
              }
              if ( (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
              {
                v13 = sub_B91C10(v10 - 24, 18);
                if ( v13 )
                {
                  if ( v15 )
                  {
                    if ( v13 != v15 )
                      return 0;
                  }
                  else
                  {
                    v15 = v13;
                  }
                }
              }
            }
LABEL_31:
            v16 += 8;
            if ( v16 == v14 )
              goto LABEL_8;
            v9 = *(_QWORD *)(*(_QWORD *)v16 + 16LL);
          }
          while ( v9 );
        }
      }
    }
  }
  return 0;
}
