// Function: sub_1C98370
// Address: 0x1c98370
//
__int64 __fastcall sub_1C98370(_QWORD *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  unsigned int v4; // r12d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rsi
  _QWORD *v11; // r10
  unsigned __int64 v12; // r11
  _QWORD *v13; // rax
  _QWORD *v14; // r9
  _QWORD *v15; // r14
  _QWORD *v16; // rbx
  __int64 v17; // r12
  unsigned int v18; // edx
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rsi
  unsigned __int8 *v25; // [rsp+18h] [rbp-88h] BYREF
  __int64 v26[5]; // [rsp+20h] [rbp-80h] BYREF
  int v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]

  if ( a1[18] )
  {
    if ( !(*(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8) )
    {
      v8 = sub_1CCDC20(a3);
      v9 = a1[16];
      v10 = v8;
      v11 = *(_QWORD **)(a1[15] + 8 * (v8 % v9));
      v12 = v8 % v9;
      if ( v11 )
      {
        v13 = (_QWORD *)*v11;
        if ( v10 == *(_QWORD *)(*v11 + 8LL) )
        {
LABEL_10:
          v15 = (_QWORD *)*v11;
          if ( *v11 )
          {
            v16 = (_QWORD *)*v15;
            if ( !*v15 )
              goto LABEL_17;
            while ( v16[1] % v9 == v12 && v10 == v16[1] )
            {
              v16 = (_QWORD *)*v16;
              if ( !v16 )
              {
LABEL_17:
                while ( 1 )
                {
                  v17 = v15[2];
                  LOBYTE(v18) = sub_15CCEE0(a1[45], v17, a2) | (a2 == v17);
                  v4 = v18;
                  if ( (_BYTE)v18 )
                    break;
                  v15 = (_QWORD *)*v15;
                  if ( v16 == v15 )
                    return 0;
                }
                v19 = sub_16498A0(a2);
                v20 = *(unsigned __int8 **)(a2 + 48);
                v26[0] = 0;
                v26[3] = v19;
                v21 = *(_QWORD *)(a2 + 40);
                v26[4] = 0;
                v26[1] = v21;
                v27 = 0;
                v28 = 0;
                v29 = 0;
                v26[2] = a2 + 24;
                v25 = v20;
                if ( !v20 )
                  goto LABEL_32;
                sub_1623A60((__int64)&v25, (__int64)v20, 2);
                if ( v26[0] )
                  sub_161E7C0((__int64)v26, v26[0]);
                v26[0] = (__int64)v25;
                if ( v25 )
                {
                  sub_1623210((__int64)&v25, v25, (__int64)v26);
                  v22 = *((_DWORD *)v15 + 6);
                  v23 = v26[0];
                  if ( v22 != 4 )
                    goto LABEL_23;
                }
                else
                {
LABEL_32:
                  v22 = *((_DWORD *)v15 + 6);
                  if ( v22 == 4 )
                  {
                    *a4 = 4;
                    return v4;
                  }
                  v23 = 0;
LABEL_23:
                  if ( v22 > 4 )
                  {
                    if ( v22 == 5 )
                    {
                      v22 = 8;
                      goto LABEL_27;
                    }
                    if ( v22 == 101 )
                    {
                      v22 = 16;
                      goto LABEL_27;
                    }
                  }
                  else
                  {
                    if ( v22 == 1 )
                      goto LABEL_27;
                    if ( v22 == 3 )
                    {
                      v22 = 2;
                      goto LABEL_27;
                    }
                  }
                  v22 = 15;
                }
LABEL_27:
                *a4 = v22;
                if ( v23 )
                  sub_161E7C0((__int64)v26, v23);
                return v4;
              }
            }
            if ( v15 != v16 )
              goto LABEL_17;
          }
        }
        else
        {
          while ( 1 )
          {
            v14 = (_QWORD *)*v13;
            if ( !*v13 )
              break;
            v11 = v13;
            if ( v12 != v14[1] % v9 )
              break;
            v13 = (_QWORD *)*v13;
            if ( v10 == v14[1] )
              goto LABEL_10;
          }
        }
      }
    }
  }
  return 0;
}
