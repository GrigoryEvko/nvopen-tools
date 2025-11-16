// Function: sub_29D84C0
// Address: 0x29d84c0
//
__int64 __fastcall sub_29D84C0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ebx
  unsigned int v4; // eax
  int v6; // eax
  __int64 *v7; // r14
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 *v10; // r15
  __int64 v11; // rax
  int v12; // r12d
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // eax
  int v16; // r12d
  __int64 *v17; // rdi
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // eax
  int v22; // r12d
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdx
  __int64 v27; // rsi
  unsigned __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // r14
  __int64 v32; // r13
  unsigned int v33; // ebx
  unsigned int v34; // eax
  unsigned __int64 v35; // [rsp+0h] [rbp-C0h]
  __int64 *v36; // [rsp+0h] [rbp-C0h]
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 *v38; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+10h] [rbp-B0h]
  __int64 *v40; // [rsp+10h] [rbp-B0h]
  int v41; // [rsp+1Ch] [rbp-A4h]
  int v43; // [rsp+48h] [rbp-78h]
  unsigned int v44; // [rsp+4Ch] [rbp-74h]
  __int64 *v45; // [rsp+50h] [rbp-70h]
  __int64 v46; // [rsp+58h] [rbp-68h]
  __int64 v47; // [rsp+60h] [rbp-60h] BYREF
  __int64 v48; // [rsp+68h] [rbp-58h] BYREF
  __int64 v49; // [rsp+70h] [rbp-50h] BYREF
  __int64 v50; // [rsp+78h] [rbp-48h] BYREF
  __int64 v51; // [rsp+80h] [rbp-40h] BYREF
  __int64 v52; // [rsp+88h] [rbp-38h] BYREF

  v48 = a2;
  v47 = a3;
  v3 = sub_A74480((__int64)&v47);
  v4 = sub_A74480((__int64)&v48);
  v44 = sub_29D7CF0((__int64)a1, v4, v3);
  if ( !v44 )
  {
    v6 = sub_A74480((__int64)&v48);
    v41 = v6 - 1;
    if ( v6 )
    {
      v43 = -1;
      v7 = &v51;
      v8 = &v52;
      while ( 1 )
      {
        v49 = sub_A74490(&v48, v43);
        v50 = sub_A74490(&v47, v43);
        v9 = (__int64 *)sub_A73280(&v49);
        v46 = sub_A73290(&v49);
        v10 = (__int64 *)sub_A73280(&v50);
        v11 = sub_A73290(&v50);
        v45 = (__int64 *)v11;
        if ( v9 != (__int64 *)v46 && v10 != (__int64 *)v11 )
        {
          do
          {
            v51 = *v9;
            v52 = *v10;
            if ( sub_A71860((__int64)v7) && sub_A71860((__int64)v8) )
            {
              v12 = sub_A71AE0(v7);
              if ( v12 != (unsigned int)sub_A71AE0(v8) )
                goto LABEL_42;
              v13 = sub_A72A60(v7);
              v14 = sub_A72A60(v8);
              if ( v13 && v14 )
              {
                v15 = sub_29D81B0(a1, v13, v14);
                if ( v15 )
                  return v15;
              }
              else
              {
                v21 = sub_29D7CF0((__int64)a1, v13, v14);
                if ( v21 )
                  return v21;
              }
            }
            else if ( sub_A71880((__int64)v7) && sub_A71880((__int64)v8) )
            {
              v16 = sub_A71AE0(v7);
              v17 = v8;
              if ( v16 != (unsigned int)sub_A71AE0(v8) )
                goto LABEL_43;
              v18 = sub_A72AA0(v8);
              v19 = sub_A72AA0(v7);
              v20 = sub_29D7DA0((__int64)a1, v19, v18);
              if ( v20 )
                return v20;
            }
            else if ( sub_A718A0((__int64)v7) && sub_A718A0((__int64)v8) )
            {
              v22 = sub_A71AE0(v7);
              if ( v22 != (unsigned int)sub_A71AE0(v8) )
              {
LABEL_42:
                v17 = v8;
LABEL_43:
                v33 = sub_A71AE0(v17);
                v34 = sub_A71AE0(v7);
                return (unsigned int)sub_29D7CF0((__int64)a1, v34, v33);
              }
              v23 = sub_A72AC0(v7);
              v25 = v24;
              v39 = v23;
              v37 = sub_A72AC0(v8);
              v35 = v26;
              v15 = sub_29D7CF0((__int64)a1, v25, v26);
              if ( v15 )
                return v15;
              v27 = v39;
              v28 = v39 + 32 * v25;
              v29 = v37 + 32 * v35;
              if ( v39 != v28 && v37 != v29 )
              {
                v40 = v9;
                v30 = v37;
                v38 = v7;
                v31 = v29;
                v36 = v8;
                v32 = v27;
                while ( 1 )
                {
                  v15 = sub_29D7DA0((__int64)a1, v32, v30);
                  if ( v15 )
                    return v15;
                  v32 += 32;
                  v30 += 32;
                  if ( v32 == v28 || v30 == v31 )
                  {
                    v9 = v40;
                    v7 = v38;
                    v8 = v36;
                    break;
                  }
                }
              }
            }
            else
            {
              if ( sub_A730F0(v7, v52) )
                return (unsigned int)-1;
              if ( sub_A730F0(v8, v51) )
                return 1;
            }
            ++v9;
            ++v10;
            if ( (__int64 *)v46 == v9 )
              goto LABEL_38;
          }
          while ( v45 != v10 );
        }
        if ( (__int64 *)v46 != v9 )
          return 1;
LABEL_38:
        if ( v45 != v10 )
          return (unsigned int)-1;
        if ( ++v43 == v41 )
          return v44;
      }
    }
  }
  return v44;
}
