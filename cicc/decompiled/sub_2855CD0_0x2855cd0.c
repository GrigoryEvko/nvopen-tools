// Function: sub_2855CD0
// Address: 0x2855cd0
//
void __fastcall sub_2855CD0(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r8
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // r13
  char v16; // di
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r9
  unsigned __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // [rsp-50h] [rbp-50h]
  unsigned __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 72) == 1 )
  {
    v2 = *(_QWORD *)(a1 + 1320);
    v22 = *(unsigned int *)(a1 + 1328);
    v3 = v2 + 2184 * v22;
    if ( v2 == v3 )
    {
      if ( (unsigned int)qword_5001308 > 1 )
        return;
    }
    else
    {
      v4 = *(_QWORD *)(a1 + 1320);
      v5 = 1;
      while ( 1 )
      {
        v6 = *(unsigned int *)(v4 + 768);
        if ( (unsigned int)v6 >= (unsigned int)qword_5001308 )
          break;
        v5 *= v6;
        if ( (unsigned int)qword_5001308 <= v5 )
          break;
        v4 += 2184;
        if ( v3 == v4 )
          return;
      }
    }
    if ( v22 )
    {
      v7 = 0;
      v8 = 0;
      while ( 1 )
      {
        v9 = v7 + v2;
        if ( *(_DWORD *)(v9 + 32) == 2
          && ((unsigned __int8)sub_DFE0F0(*(_QWORD *)(a1 + 48)) || (unsigned __int8)sub_DFE120(*(_QWORD *)(a1 + 48))) )
        {
          v10 = *(unsigned int *)(v9 + 768);
          v11 = *(_QWORD *)(v9 + 760);
          v12 = v11 + 112 * v10;
          if ( v11 == v12 )
          {
            v14 = -1;
          }
          else
          {
            v13 = *(_QWORD *)(v9 + 760);
            v14 = -1;
            do
            {
              if ( v14 > *(unsigned int *)(v13 + 48) - ((unsigned __int64)(*(_QWORD *)(v13 + 88) == 0) - 1) )
                v14 = *(unsigned int *)(v13 + 48) - ((*(_QWORD *)(v13 + 88) == 0) - 1LL);
              v13 += 112;
            }
            while ( v12 != v13 );
          }
          v15 = 0;
          v16 = 0;
          if ( *(_DWORD *)(v9 + 768) )
          {
            while ( 1 )
            {
              v17 = v11 + 112 * v15;
              if ( *(unsigned int *)(v17 + 48) - ((unsigned __int64)(*(_QWORD *)(v17 + 88) == 0) - 1) <= v14 )
              {
                if ( v10 == ++v15 )
                  goto LABEL_25;
              }
              else
              {
                v23 = v14;
                v24 = v10;
                sub_28532A0(v9, (__int64 *)v17);
                v14 = v23;
                v16 = 1;
                v10 = v24 - 1;
                if ( v24 - 1 == v15 )
                {
LABEL_25:
                  if ( v16 )
                    sub_2855860(v9, v8, a1 + 36280);
                  break;
                }
              }
              v11 = *(_QWORD *)(v9 + 760);
            }
          }
          v18 = *(_QWORD *)(a1 + 1320);
          v19 = v18 + 2184LL * *(unsigned int *)(a1 + 1328);
          if ( v18 == v19 )
          {
            if ( (unsigned int)qword_5001308 > 1 )
              return;
          }
          else
          {
            v20 = 1;
            while ( 1 )
            {
              v21 = *(unsigned int *)(v18 + 768);
              if ( (unsigned int)v21 >= (unsigned int)qword_5001308 )
                break;
              v20 *= v21;
              if ( (unsigned int)qword_5001308 <= v20 )
                break;
              v18 += 2184;
              if ( v19 == v18 )
                return;
            }
          }
        }
        ++v8;
        v7 += 2184;
        if ( v8 == v22 )
          return;
        v2 = *(_QWORD *)(a1 + 1320);
      }
    }
  }
}
