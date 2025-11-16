// Function: sub_31FBCA0
// Address: 0x31fbca0
//
void __fastcall sub_31FBCA0(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int8 v3; // dl
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int64 *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 v11; // al
  __int64 v12; // r12
  unsigned __int8 v13; // dl
  unsigned __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  const char *v17; // rax
  char *v18; // rdx
  char *v19; // r8
  char *v20; // rcx
  unsigned __int8 v21; // al
  unsigned __int64 *v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // eax
  char *v25; // rdx
  unsigned __int64 *v26; // [rsp+8h] [rbp-A8h] BYREF
  __m128i v27; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+20h] [rbp-90h] BYREF
  _BYTE *v29; // [rsp+30h] [rbp-80h] BYREF
  __int64 v30; // [rsp+38h] [rbp-78h]
  _BYTE v31[112]; // [rsp+40h] [rbp-70h] BYREF

  v26 = a2;
  v3 = *((_BYTE *)a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(a2 - 4);
  else
    v4 = (__int64)&a2[-((v3 >> 2) & 0xF) - 2];
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5 )
  {
    sub_B91420(v5);
    if ( v6 )
    {
      v7 = v26;
      if ( v26 )
      {
        if ( (unsigned __int16)sub_AF18C0((__int64)v26) != 22
          || ((v21 = *((_BYTE *)v26 - 16), (v21 & 2) == 0)
            ? (v22 = &v26[-((v21 >> 2) & 0xF) - 2])
            : (v22 = (unsigned __int64 *)*(v26 - 4)),
              (v23 = v22[1]) == 0
           || (v24 = sub_AF18C0(v23), v8 = v24, (unsigned __int16)v24 > 0x17u)
           || ((1LL << v24) & 0x880004) == 0) )
        {
          while ( (*((_BYTE *)v7 + 20) & 4) == 0 )
          {
            if ( *(_BYTE *)v7 != 13 )
            {
              v30 = 0x500000000LL;
              v29 = v31;
              v13 = *((_BYTE *)v26 - 16);
              if ( (v13 & 2) != 0 )
                v14 = (unsigned __int64 *)*(v26 - 4);
              else
                v14 = &v26[-((v13 >> 2) & 0xF) - 2];
              v15 = v14[1];
              v16 = sub_31F7970((__int64)a1, v15, (__int64)&v29, v8, v9, v10);
              v17 = sub_AF5A10((unsigned __int8 *)v26, v15);
              v19 = v18;
              v20 = (char *)v17;
              if ( !v18 )
              {
                v20 = (char *)sub_31F3D90((__int64)v26);
                v19 = v25;
              }
              sub_31F5640((__int64)&v27, (__int64)v29, (unsigned int)v30, v20, v19);
              if ( v16 )
              {
                if ( (unsigned __int8 *)a1[167] == v16 )
                  sub_31FBC20(a1 + 168, &v27, &v26);
              }
              else
              {
                sub_31FBC20(a1 + 171, &v27, &v26);
              }
              if ( (__int64 *)v27.m128i_i64[0] != &v28 )
                j_j___libc_free_0(v27.m128i_u64[0]);
              if ( v29 != v31 )
                _libc_free((unsigned __int64)v29);
              return;
            }
            v11 = *((_BYTE *)v7 - 16);
            if ( (v11 & 2) != 0 )
              v12 = *(v7 - 4);
            else
              v12 = (__int64)&v7[-((v11 >> 2) & 0xF) - 2];
            v7 = *(unsigned __int64 **)(v12 + 24);
            if ( !v7 )
              return;
          }
        }
      }
    }
  }
}
