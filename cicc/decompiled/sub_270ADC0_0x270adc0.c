// Function: sub_270ADC0
// Address: 0x270adc0
//
char __fastcall sub_270ADC0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // r10
  char *v8; // r14
  __int64 v9; // r11
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 i; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned int v24; // eax
  unsigned __int64 v25; // r8
  char *v26; // rbx
  __int64 v28; // [rsp+0h] [rbp-80h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+18h] [rbp-68h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  _BYTE v35[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v36; // [rsp+40h] [rbp-40h]

  LOBYTE(v3) = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 22 || (unsigned __int8)v3 > 0x1Cu )
  {
    v4 = *(_QWORD *)(a2 + 16);
    v3 = (__int64 *)v35;
    while ( v4 )
    {
      v5 = *(_QWORD *)(v4 + 8);
      v6 = (unsigned int)sub_BD2910(v4);
      LOBYTE(v3) = sub_B1A070(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL), v4);
      if ( (_BYTE)v3 )
      {
        LOBYTE(v3) = sub_B19F20(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL), *(char **)a1, v4);
        if ( (_BYTE)v3 )
        {
          **(_BYTE **)(a1 + 8) = 1;
          v7 = *(_QWORD *)(v4 + 24);
          v8 = *(char **)a1;
          v9 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
          if ( *(_BYTE *)v7 == 84 )
          {
            v3 = (__int64 *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8 * v6);
            v12 = *v3;
            if ( *((_QWORD *)v8 + 1) != v9 )
            {
              for ( i = *v3; ; i = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v23) + 8LL) )
              {
                v30 = v7;
                v33 = v9;
                v21 = sub_AA4FF0(i);
                if ( !v21 )
                  BUG();
                v9 = v33;
                v7 = v30;
                if ( *(_BYTE *)(v21 - 24) != 39 )
                  break;
                v22 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL);
                if ( i )
                {
                  v23 = (unsigned int)(*(_DWORD *)(i + 44) + 1);
                  v24 = *(_DWORD *)(i + 44) + 1;
                }
                else
                {
                  v23 = 0;
                  v24 = 0;
                }
                if ( v24 >= *(_DWORD *)(v22 + 32) )
                  BUG();
              }
              v34 = v30;
              v36 = 257;
              v28 = v9;
              v25 = (*(_QWORD *)(i + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24;
              if ( (*(_QWORD *)(i + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                v25 = 0;
              v31 = v25 + 24;
              v3 = sub_BD2C40(72, unk_3F10A14);
              v7 = v34;
              v26 = (char *)v3;
              if ( v3 )
              {
                LOBYTE(v3) = sub_B51BF0((__int64)v3, (__int64)v8, v28, (__int64)v35, v31, 0);
                v7 = v34;
              }
              v8 = v26;
            }
            if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
            {
              v4 = v5;
              v13 = 0;
              v14 = 8LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
              do
              {
                v15 = *(_QWORD *)(v7 - 8);
                v3 = (__int64 *)(v15 + 32LL * *(unsigned int *)(v7 + 72));
                if ( v12 == v3[v13 / 8] )
                {
                  v16 = 4 * v13;
                  if ( v4 )
                  {
                    v17 = *(_QWORD *)(v7 - 8);
                    if ( (*(_BYTE *)(v7 + 7) & 0x40) == 0 )
                      v17 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
                    if ( v4 == v16 + v17 )
                      v4 = *(_QWORD *)(v4 + 8);
                  }
                  v3 = (__int64 *)(v15 + v16);
                  if ( *v3 )
                  {
                    v18 = v3[1];
                    *(_QWORD *)v3[2] = v18;
                    if ( v18 )
                      *(_QWORD *)(v18 + 16) = v3[2];
                  }
                  *v3 = (__int64)v8;
                  if ( v8 )
                  {
                    v19 = *((_QWORD *)v8 + 2);
                    v3[1] = v19;
                    if ( v19 )
                      *(_QWORD *)(v19 + 16) = v3 + 1;
                    v3[2] = (__int64)(v8 + 16);
                    *((_QWORD *)v8 + 2) = v3;
                  }
                }
                v13 += 8LL;
              }
              while ( v14 != v13 );
              continue;
            }
          }
          else
          {
            v32 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
            if ( v9 == *((_QWORD *)v8 + 1) )
              goto LABEL_10;
            v36 = 257;
            v29 = *(_QWORD *)(v4 + 24) + 24LL;
            v3 = sub_BD2C40(72, unk_3F10A14);
            if ( v3 )
            {
              v10 = (__int64)v8;
              v8 = (char *)v3;
              sub_B51BF0((__int64)v3, v10, v32, (__int64)v35, v29, 0);
              if ( *(_QWORD *)v4 )
              {
LABEL_10:
                v11 = *(_QWORD *)(v4 + 8);
                **(_QWORD **)(v4 + 16) = v11;
                if ( v11 )
                  *(_QWORD *)(v11 + 16) = *(_QWORD *)(v4 + 16);
              }
              *(_QWORD *)v4 = v8;
              v3 = (__int64 *)*((_QWORD *)v8 + 2);
              *(_QWORD *)(v4 + 8) = v3;
              if ( v3 )
                v3[2] = v4 + 8;
              *(_QWORD *)(v4 + 16) = v8 + 16;
              *((_QWORD *)v8 + 2) = v4;
              v4 = v5;
              continue;
            }
            if ( *(_QWORD *)v4 )
            {
              v3 = *(__int64 **)(v4 + 8);
              **(_QWORD **)(v4 + 16) = v3;
              if ( v3 )
                v3[2] = *(_QWORD *)(v4 + 16);
              *(_QWORD *)v4 = 0;
              v4 = v5;
              continue;
            }
          }
        }
      }
      v4 = v5;
    }
  }
  return (char)v3;
}
