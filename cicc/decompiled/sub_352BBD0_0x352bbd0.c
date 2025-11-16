// Function: sub_352BBD0
// Address: 0x352bbd0
//
char __fastcall sub_352BBD0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rax
  __int64 v4; // r14
  int v5; // eax
  const char *v6; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v9)(_QWORD *, _QWORD *, __int64); // [rsp+10h] [rbp-60h]
  const char *v10; // [rsp+20h] [rbp-50h] BYREF
  char v11; // [rsp+40h] [rbp-30h]
  char v12; // [rsp+41h] [rbp-2Fh]

  v2 = sub_352B010(a2);
  v3 = sub_352B6D0(a1, a2);
  v4 = v3;
  if ( v2 == 1 )
  {
    if ( !(unsigned __int8)sub_352B040(a2) )
    {
      sub_2EE7320(v8, a1 + 144, a2);
      v12 = 1;
      v6 = "Entry intrinsic can occur only in a convergent function.";
      goto LABEL_25;
    }
    if ( sub_2E31AB0(*(_QWORD *)(a2 + 24)) )
    {
      if ( !*(_BYTE *)(a1 + 192) )
        goto LABEL_15;
      sub_2EE7320(v8, a1 + 144, a2);
      v12 = 1;
      v6 = "Entry intrinsic cannot be preceded by a convergent operation in the same basic block.";
    }
    else
    {
      sub_2EE7320(v8, a1 + 144, a2);
      v12 = 1;
      v6 = "Entry intrinsic can occur only in the entry block.";
    }
  }
  else
  {
    if ( v2 != 2 )
    {
      if ( v2 )
      {
        if ( v2 == 3 )
        {
          if ( !sub_352B050(a2) )
          {
LABEL_6:
            if ( !v4 && v2 == 3 )
            {
              LOBYTE(v5) = sub_352B050(a2);
              if ( !(_BYTE)v5 )
                return v5;
              v5 = *(_DWORD *)(a1 + 152);
              if ( v5 )
              {
                *(_DWORD *)(a1 + 152) = 1;
                return v5;
              }
LABEL_29:
              sub_2EE7320(v8, a1 + 144, a2);
              v12 = 1;
              v6 = "Cannot mix controlled and uncontrolled convergence in the same function.";
              goto LABEL_25;
            }
LABEL_18:
            LOBYTE(v5) = sub_352B050(a2);
            if ( !(_BYTE)v5 )
            {
              sub_2EE7320(v8, a1 + 144, a2);
              v12 = 1;
              v6 = "Convergence control token can only be used in a convergent call.";
              goto LABEL_25;
            }
            if ( *(_DWORD *)(a1 + 152) != 1 )
            {
              *(_DWORD *)(a1 + 152) = 0;
              return v5;
            }
            goto LABEL_29;
          }
LABEL_17:
          *(_BYTE *)(a1 + 192) = 1;
          goto LABEL_6;
        }
LABEL_16:
        sub_352B3E0(a1, a2);
        if ( !sub_352B050(a2) )
          goto LABEL_18;
        goto LABEL_17;
      }
LABEL_15:
      if ( v4 )
      {
        sub_2EE7320(v8, a1 + 144, a2);
        v12 = 1;
        v6 = "Entry or anchor intrinsic cannot have a convergencectrl token operand.";
        goto LABEL_25;
      }
      goto LABEL_16;
    }
    if ( v3 )
    {
      if ( *(_BYTE *)(a1 + 192) )
      {
        sub_2EE7320(v8, a1 + 144, a2);
        v12 = 1;
        v6 = "Loop intrinsic cannot be preceded by a convergent operation in the same basic block.";
        goto LABEL_25;
      }
      goto LABEL_16;
    }
    sub_2EE7320(v8, a1 + 144, a2);
    v12 = 1;
    v6 = "Loop intrinsic must have a convergencectrl token operand.";
  }
LABEL_25:
  v10 = v6;
  v11 = 3;
  sub_352B2E0((_BYTE *)a1, (__int64)&v10, v8, 1);
  LOBYTE(v5) = (_BYTE)v9;
  if ( v9 )
    LOBYTE(v5) = v9(v8, v8, 3);
  return v5;
}
