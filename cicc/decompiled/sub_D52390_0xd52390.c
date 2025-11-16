// Function: sub_D52390
// Address: 0xd52390
//
__int64 __fastcall sub_D52390(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 *v16; // rcx
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  int v22; // [rsp+18h] [rbp-58h]
  char v23; // [rsp+1Ch] [rbp-54h]
  char v24; // [rsp+20h] [rbp-50h] BYREF

  v3 = a1;
  if ( a1 != a2 )
  {
    v4 = a2;
    if ( sub_AA5780(a1) )
    {
      v19 = 0;
      v20 = (__int64 *)&v24;
      v21 = 4;
      v22 = 0;
      v23 = 1;
      v6 = sub_AA5780(a1);
      v9 = v6;
      if ( a2 != v6 )
      {
        if ( v6 )
        {
          v10 = *(_QWORD *)(v6 + 56);
          v11 = v9 + 48;
          if ( v9 + 48 != v10 )
          {
            while ( 1 )
            {
              v12 = 0;
              do
              {
                v10 = *(_QWORD *)(v10 + 8);
                ++v12;
              }
              while ( v11 != v10 );
              v13 = v23;
              if ( v12 != 1 )
                goto LABEL_29;
              if ( !v23 )
                break;
              v14 = HIDWORD(v21);
              v15 = v20;
              v16 = &v20[HIDWORD(v21)];
              a2 = HIDWORD(v21);
              if ( v20 == v16 )
              {
                if ( !a3 )
                {
LABEL_35:
                  if ( (unsigned int)v21 > (unsigned int)a2 )
                  {
                    a2 = (unsigned int)(a2 + 1);
                    HIDWORD(v21) = a2;
                    *v16 = v9;
                    ++v19;
                    goto LABEL_25;
                  }
                  goto LABEL_33;
                }
LABEL_19:
                if ( !sub_AA5510(v9) )
                  goto LABEL_28;
LABEL_20:
                if ( v23 )
                {
                  v14 = HIDWORD(v21);
                  v15 = v20;
                  v16 = &v20[HIDWORD(v21)];
                  a2 = HIDWORD(v21);
                  if ( v20 != v16 )
                    goto LABEL_24;
                  goto LABEL_35;
                }
LABEL_33:
                a2 = v9;
                sub_C8CC70((__int64)&v19, v9, v14, (__int64)v16, v7, v8);
                goto LABEL_25;
              }
              v14 = (__int64)v20;
              do
              {
                if ( *(_QWORD *)v14 == v9 )
                {
                  if ( v4 == v9 )
                    return v4;
                  return v3;
                }
                v14 += 8;
              }
              while ( v16 != (__int64 *)v14 );
              if ( a3 )
                goto LABEL_19;
LABEL_24:
              while ( v9 != *v15 )
              {
                if ( ++v15 == v16 )
                  goto LABEL_35;
              }
LABEL_25:
              v3 = v9;
              v18 = sub_AA5780(v9);
              if ( !v18 || v4 == v18 )
              {
                v3 = v9;
                v9 = v18;
                goto LABEL_28;
              }
              v9 = v18;
              v10 = *(_QWORD *)(v18 + 56);
              v11 = v9 + 48;
              if ( v9 + 48 == v10 )
                goto LABEL_28;
            }
            a2 = v9;
            if ( sub_C8CA60((__int64)&v19, v9) )
              goto LABEL_28;
            if ( a3 )
              goto LABEL_19;
            goto LABEL_20;
          }
        }
      }
LABEL_28:
      v13 = v23;
LABEL_29:
      if ( v4 == v9 )
        v3 = v4;
      if ( !v13 )
        _libc_free(v20, a2);
    }
  }
  return v3;
}
