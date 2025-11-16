// Function: sub_269BCA0
// Address: 0x269bca0
//
void __fastcall sub_269BCA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v5; // rdx
  unsigned __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 i; // r15
  _QWORD *v9; // rdi
  __int64 *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int8 *v13; // rdi
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 *v24; // [rsp+10h] [rbp-A0h]
  __int64 *v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+18h] [rbp-98h]
  __int64 v27; // [rsp+28h] [rbp-88h] BYREF
  _BYTE v28[16]; // [rsp+30h] [rbp-80h] BYREF
  __int64 (__fastcall *v29)(_BYTE *, __int64, int); // [rsp+40h] [rbp-70h]
  __int64 (*v30)(); // [rsp+48h] [rbp-68h]
  __m128i v31[2]; // [rsp+50h] [rbp-60h] BYREF
  char v32; // [rsp+70h] [rbp-40h]

  if ( byte_4FF4EA8 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return;
  }
  v3 = *(_QWORD *)(a2 + 208);
  v5 = *(_QWORD *)(v3 + 32432);
  if ( v5 )
  {
    v30 = sub_266DEE0;
    v29 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_266DF00;
    v6 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    {
      v6 = *(_QWORD *)(v6 + 24);
      v7 = *(_BYTE *)v6;
      if ( !*(_BYTE *)v6 )
        goto LABEL_5;
    }
    else
    {
      v7 = *(_BYTE *)v6;
      if ( !*(_BYTE *)v6 )
      {
LABEL_5:
        for ( i = *(_QWORD *)(v5 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v13 = *(unsigned __int8 **)(i + 24);
          v14 = *v13;
          if ( (unsigned __int8)v14 > 0x1Cu )
          {
            v15 = (unsigned int)(v14 - 34);
            if ( (unsigned __int8)v15 <= 0x33u )
            {
              v16 = 0x8000000000041LL;
              if ( _bittest64(&v16, v15) )
              {
                v27 = *(_QWORD *)(i + 24);
                if ( v6 == sub_B43CB0((__int64)v13) )
                {
                  if ( *(_DWORD *)(a1 + 120) )
                  {
                    sub_269BA20((__int64)v31, a1 + 104, &v27);
                    if ( v32 )
                    {
                      v19 = *(unsigned int *)(a1 + 144);
                      v20 = v27;
                      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 148) )
                      {
                        v26 = v27;
                        sub_C8D5F0(a1 + 136, (const void *)(a1 + 152), v19 + 1, 8u, v27, v18);
                        v19 = *(unsigned int *)(a1 + 144);
                        v20 = v26;
                      }
                      *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v19) = v20;
                      ++*(_DWORD *)(a1 + 144);
                    }
                  }
                  else
                  {
                    v9 = *(_QWORD **)(a1 + 136);
                    v10 = &v9[*(unsigned int *)(a1 + 144)];
                    if ( v10 == sub_266E4D0(v9, (__int64)v10, &v27) )
                    {
                      v21 = v27;
                      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 148) )
                      {
                        sub_C8D5F0(a1 + 136, (const void *)(a1 + 152), v12 + 1, 8u, v11, v12);
                        v10 = (__int64 *)(*(_QWORD *)(a1 + 136) + 8LL * *(unsigned int *)(a1 + 144));
                      }
                      *v10 = v21;
                      v22 = (unsigned int)(*(_DWORD *)(a1 + 144) + 1);
                      *(_DWORD *)(a1 + 144) = v22;
                      if ( (unsigned int)v22 > 4 )
                      {
                        v23 = *(__int64 **)(a1 + 136);
                        v24 = &v23[v22];
                        do
                        {
                          v25 = v23;
                          sub_269BA20((__int64)v31, a1 + 104, v23);
                          v23 = v25 + 1;
                        }
                        while ( v24 != v25 + 1 );
                      }
                    }
                  }
                  v31[0].m128i_i64[1] = 0;
                  v31[0].m128i_i64[0] = v27 & 0xFFFFFFFFFFFFFFFCLL | 1;
                  nullsub_1518();
                  sub_267E550(a2, v31, (__int64)v28);
                }
              }
            }
          }
        }
        sub_2673100(a1, a2);
        if ( v29 )
          v29(v28, (__int64)v28, 3);
        return;
      }
    }
    if ( v7 == 22 )
    {
      v6 = *(_QWORD *)(v6 + 24);
    }
    else if ( v7 <= 0x1Cu )
    {
      v6 = 0;
    }
    else
    {
      v17 = sub_B43CB0(v6);
      v5 = *(_QWORD *)(v3 + 32432);
      v6 = v17;
    }
    goto LABEL_5;
  }
}
