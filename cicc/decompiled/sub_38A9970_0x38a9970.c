// Function: sub_38A9970
// Address: 0x38a9970
//
__int64 __fastcall sub_38A9970(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r12
  unsigned __int64 v9; // rsi
  const char *v10; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+10h] [rbp-30h]
  char v12; // [rsp+11h] [rbp-2Fh]

  v6 = a1 + 72;
  if ( !sub_2241AC0(a1 + 72, "DILocation") )
    return sub_38A2A90(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIExpression") )
    return sub_388E560(a1, a2, a3);
  if ( !sub_2241AC0(v6, "DIGlobalVariableExpression") )
    return sub_38A89F0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "GenericDINode") )
    return sub_38A2460(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DISubrange") )
    return sub_38A9450(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIEnumerator") )
    return sub_388D380(a1, a2, a3);
  if ( !sub_2241AC0(v6, "DIBasicType") )
    return sub_388D8B0(a1, a2, a3);
  if ( !sub_2241AC0(v6, "DIDerivedType") )
    return sub_38A35C0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DICompositeType") )
    return sub_38A3D00(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DISubroutineType") )
    return sub_38A4B10(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIFile") )
    return sub_388DC10(a1, a2, a3);
  if ( !sub_2241AC0(v6, "DICompileUnit") )
    return sub_38A5820(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DISubprogram") )
    return sub_38A5860(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DILexicalBlock") )
    return sub_38A63B0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DILexicalBlockFile") )
    return sub_38A66F0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DINamespace") )
    return sub_38A6D50(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIModule") )
    return sub_38A7350(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DITemplateTypeParameter") )
    return sub_38A76D0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DITemplateValueParameter") )
    return sub_38A7920(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIGlobalVariable") )
    return sub_38A7C60(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DILocalVariable") )
    return sub_38A81E0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DILabel") )
    return sub_38A8660(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIObjCProperty") )
    return sub_38A8C60(a1, (unsigned int **)a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIImportedEntity") )
    return sub_38A9050(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIMacro") )
    return sub_388E200(a1, a2, a3);
  if ( !sub_2241AC0(v6, "DIMacroFile") )
    return sub_38A7000(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DICommonBlock") )
    return sub_38A69D0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIStringType") )
    return sub_38A31B0(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIFortranArrayType") )
    return sub_38A4580(a1, a2, a3, a4, a5, a6);
  if ( !sub_2241AC0(v6, "DIFortranSubrange") )
    return sub_38A2DD0(a1, a2, a3, a4, a5, a6);
  v9 = *(_QWORD *)(a1 + 56);
  v12 = 1;
  v11 = 3;
  v10 = "expected metadata type";
  return sub_38814C0(a1 + 8, v9, (__int64)&v10);
}
