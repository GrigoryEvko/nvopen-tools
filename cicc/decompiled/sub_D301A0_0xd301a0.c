// Function: sub_D301A0
// Address: 0xd301a0
//
__int64 __fastcall sub_D301A0(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int8 a4, _BYTE *a5, _BYTE *a6)
{
  _BYTE *v8; // r12
  unsigned __int8 v10; // al
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rax
  char v13; // al
  __int64 v14; // r8
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rdx
  unsigned __int8 *v23; // rax
  __int64 v24; // rax
  bool v25; // dl
  bool v26; // r12
  unsigned int v27; // edx
  __int64 *v28; // rax
  char v29; // al
  unsigned __int64 *v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  unsigned __int64 v36; // [rsp+20h] [rbp-60h]
  unsigned __int64 v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h]
  unsigned __int64 v39; // [rsp+40h] [rbp-40h] BYREF
  __int64 v40; // [rsp+48h] [rbp-38h]

  v8 = a2;
  v10 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 61 )
  {
    if ( (unsigned __int8)sub_B46500((unsigned __int8 *)a1) < a4 )
      return 0;
    v11 = sub_BD3990(*(unsigned __int8 **)(a1 - 32), (__int64)a2);
    if ( !(unsigned __int8)sub_D2F730(v11, a2) )
      return 0;
    a2 = (_BYTE *)a3;
    if ( (unsigned __int8)sub_B50C50(*(_QWORD *)(a1 + 8), a3, (__int64)a5) )
    {
      v14 = a1;
      if ( a6 )
        *a6 = 1;
      return v14;
    }
    v10 = *(_BYTE *)a1;
  }
  if ( v10 == 62 )
  {
    if ( (unsigned __int8)sub_B46500((unsigned __int8 *)a1) < a4 )
      return 0;
    v12 = sub_BD3990(*(unsigned __int8 **)(a1 - 32), (__int64)a2);
    if ( !(unsigned __int8)sub_D2F730(v12, v8) )
      return 0;
    if ( a6 )
      *a6 = 0;
    v31 = *(_QWORD *)(a1 - 64);
    v13 = sub_B50C50(*(_QWORD *)(v31 + 8), a3, (__int64)a5);
    v14 = v31;
    if ( v13 )
      return v14;
    a2 = (_BYTE *)a3;
    v37 = sub_9208B0((__int64)a5, *(_QWORD *)(v31 + 8));
    v38 = v15;
    v16 = sub_9208B0((__int64)a5, a3);
    v40 = v17;
    v39 = v16;
    if ( v37 >= v16 && *(_BYTE *)v31 <= 0x15u )
      return sub_9717D0(v31, a3, a5);
    v10 = *(_BYTE *)a1;
  }
  if ( v10 == 85 )
  {
    v19 = *(_QWORD *)(a1 - 32);
    if ( v19 )
    {
      if ( !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
      {
        v14 = 0;
        if ( ((*(_DWORD *)(v19 + 36) - 243) & 0xFFFFFFFD) != 0 )
          return v14;
        if ( !a4 )
        {
          v20 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
          v21 = *(_QWORD *)(a1 + 32 * (1 - v20));
          v22 = 32 * (2 - v20);
          if ( *(_BYTE *)v21 != 17 )
            return v14;
          if ( **(_BYTE **)(a1 + v22) == 17 )
          {
            v32 = *(_QWORD *)(a1 + v22);
            v23 = sub_BD3990(*(unsigned __int8 **)(a1 - 32 * v20), (__int64)a2);
            if ( (unsigned __int8)sub_D2F730(v23, v8) )
            {
              if ( a6 )
                *a6 = 0;
              v24 = sub_9208B0((__int64)a5, a3);
              v26 = v25;
              v36 = v24;
              if ( !v25 )
              {
                LODWORD(v38) = *(_DWORD *)(v32 + 32);
                if ( (unsigned int)v38 > 0x40 )
                  sub_C43780((__int64)&v37, (const void **)(v32 + 24));
                else
                  v37 = *(_QWORD *)(v32 + 24);
                sub_C47170((__int64)&v37, 8u);
                v27 = v38;
                LODWORD(v38) = 0;
                LODWORD(v40) = v27;
                v39 = v37;
                if ( v27 <= 0x40 )
                {
                  v26 = v37 < v36;
                }
                else
                {
                  v30 = (unsigned __int64 *)v37;
                  if ( v27 - (unsigned int)sub_C444A0((__int64)&v39) <= 0x40 )
                    v26 = v36 > *v30;
                  if ( v30 )
                  {
                    j_j___libc_free_0_0(v30);
                    if ( (unsigned int)v38 > 0x40 )
                    {
                      if ( v37 )
                        j_j___libc_free_0_0(v37);
                    }
                  }
                }
                if ( !v26 )
                {
                  if ( v36 <= 7 )
                    sub_C44740((__int64)&v39, (char **)(v21 + 24), v36);
                  else
                    sub_C47700((__int64)&v39, v36, v21 + 24);
                  v28 = (__int64 *)sub_BD5C60(a1);
                  v34 = sub_ACCFD0(v28, (__int64)&v39);
                  v29 = sub_B50C50(*(_QWORD *)(v34 + 8), a3, (__int64)a5);
                  v14 = v34;
                  if ( !v29 )
                    v14 = 0;
                  if ( (unsigned int)v40 > 0x40 )
                  {
                    if ( v39 )
                    {
                      v35 = v14;
                      j_j___libc_free_0_0(v39);
                      return v35;
                    }
                  }
                  return v14;
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
