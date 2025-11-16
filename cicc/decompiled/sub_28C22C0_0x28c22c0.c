// Function: sub_28C22C0
// Address: 0x28c22c0
//
__int64 __fastcall sub_28C22C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 *v14; // rdi
  __int64 *v15; // rdx
  __int64 *v16; // rax
  __int64 *v17; // r8
  __int64 *v18; // rdi
  unsigned __int8 *v19; // rbx
  unsigned __int8 *v20; // rdx
  __int64 *v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // r14
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 *v27; // rax
  _QWORD *v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 *v34; // [rsp+10h] [rbp-A0h]
  __int64 v35; // [rsp+10h] [rbp-A0h]
  __int64 *v36; // [rsp+18h] [rbp-98h]
  __int64 *v37; // [rsp+18h] [rbp-98h]
  __int64 *v38; // [rsp+18h] [rbp-98h]
  __int64 *v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+28h] [rbp-88h] BYREF
  __int64 v42; // [rsp+30h] [rbp-80h] BYREF
  __int64 v43; // [rsp+38h] [rbp-78h] BYREF
  _QWORD *v44; // [rsp+40h] [rbp-70h] BYREF
  const void **v45; // [rsp+48h] [rbp-68h] BYREF
  const char *v46; // [rsp+50h] [rbp-60h] BYREF
  __int64 v47; // [rsp+58h] [rbp-58h]
  __int64 *v48; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v49; // [rsp+68h] [rbp-48h]
  __int16 v50; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 16);
  if ( v4 )
  {
    v4 = *(_QWORD *)(v4 + 8);
    if ( v4 )
    {
      return 0;
    }
    else
    {
      v41 = 0;
      v42 = 0;
      if ( (unsigned __int8)sub_28C1E30(a1, (unsigned __int8 *)a4, (_BYTE *)a2, &v41, &v42) )
      {
        v39 = sub_DD8400(*(_QWORD *)(a1 + 24), v41);
        v21 = sub_DD8400(*(_QWORD *)(a1 + 24), v42);
        v22 = sub_DD8400(*(_QWORD *)(a1 + 24), a3);
        v23 = (__int64)v22;
        if ( v21 == v22
          || (v35 = v42,
              v24 = sub_28C1E80(a1, (unsigned __int8 *)a4, (__int64)v39, (__int64)v22),
              (v25 = sub_28C2170(a1, (__int64)v24, v35, (unsigned __int8 *)a4)) == 0) )
        {
          if ( v39 != (__int64 *)v23 )
          {
            v26 = v41;
            v27 = sub_28C1E80(a1, (unsigned __int8 *)a4, (__int64)v21, v23);
            return sub_28C2170(a1, (__int64)v27, v26, (unsigned __int8 *)a4);
          }
        }
        else
        {
          return v25;
        }
      }
      else
      {
        v43 = 0;
        v44 = 0;
        v45 = 0;
        if ( sub_28C1CF0(a1, (_BYTE *)a4, (_BYTE *)a2, a3, &v43, &v44, &v45) )
        {
          v28 = v44;
          LODWORD(v47) = *((_DWORD *)v45 + 2);
          if ( (unsigned int)v47 > 0x40 )
            sub_C43780((__int64)&v46, v45);
          else
            v46 = (const char *)*v45;
          sub_C48380((__int64)&v46, (__int64)v28);
          v29 = sub_AD8D80(*(_QWORD *)(a4 + 8), (__int64)&v46);
          v30 = v29;
          if ( (unsigned int)v47 > 0x40 && v46 )
          {
            v40 = v29;
            j_j___libc_free_0_0((unsigned __int64)v46);
            v30 = v40;
          }
          v46 = "add.nary";
          v50 = 259;
          v31 = sub_B504D0(13, v43, v30, (__int64)&v46, a4 + 24, 0);
          v32 = sub_AD8D80(*(_QWORD *)(a4 + 8), (__int64)v44);
          v50 = 259;
          v46 = "shl.nary";
          return sub_B504D0(25, v31, v32, (__int64)&v46, a4 + 24, 0);
        }
        else if ( *(_BYTE *)a4 == 46 && *(_BYTE *)a2 == 42 )
        {
          v10 = *(_QWORD *)(a2 - 64);
          if ( v10 )
          {
            v11 = *(_QWORD *)(a2 - 32);
            v41 = *(_QWORD *)(a2 - 64);
            if ( v11 )
            {
              v12 = *(_QWORD *)(a1 + 24);
              v42 = v11;
              v36 = sub_DD8400(v12, v10);
              v34 = sub_DD8400(*(_QWORD *)(a1 + 24), v42);
              v13 = sub_DD8400(*(_QWORD *)(a1 + 24), a3);
              v14 = *(__int64 **)(a1 + 24);
              v15 = v36;
              v49 = v13;
              v37 = v13;
              v48 = v15;
              v46 = (const char *)&v48;
              v47 = 0x200000002LL;
              v16 = sub_DC8BD0(v14, (__int64)&v46, 0, 0);
              v17 = v37;
              v33 = (__int64)v16;
              if ( v46 != (const char *)&v48 )
              {
                _libc_free((unsigned __int64)v46);
                v17 = v37;
              }
              v18 = *(__int64 **)(a1 + 24);
              v46 = (const char *)&v48;
              v48 = v34;
              v49 = v17;
              v47 = 0x200000002LL;
              v38 = sub_DC8BD0(v18, (__int64)&v46, 0, 0);
              if ( v46 != (const char *)&v48 )
                _libc_free((unsigned __int64)v46);
              v19 = sub_28C1F40(a1, v33, a4);
              if ( v19 )
              {
                v20 = sub_28C1F40(a1, (__int64)v38, a4);
                if ( v20 )
                {
                  v50 = 257;
                  v4 = sub_B504D0(13, (__int64)v19, (__int64)v20, (__int64)&v46, a4 + 24, 0);
                  sub_BD6B90((unsigned __int8 *)v4, (unsigned __int8 *)a4);
                }
              }
            }
          }
        }
      }
    }
  }
  return v4;
}
